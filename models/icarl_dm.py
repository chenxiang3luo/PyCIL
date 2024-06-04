import copy
import logging
import math
import os
import pickle
import time
from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import torch
from dd_algorithms.dm import DistributionMatching
from dd_algorithms.utils import (DiffAugment, EpisodicTensorDataset,
                                 ParamDiffAug, get_time, save_images)
from models.base import BaseLearner
from PIL import Image
from utils.data_manager import pil_loader
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import (denormalize_cifar100, denormalize_imageNet,target2onehot, tensor2img,
                           tensor2numpy)
from utils.data_manager import DummyDataset,RepeatSampler

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.01
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170
# epochs = 2
lrate = 0.01
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
#weight_decay = 0.0005
weight_decay =2e-4
num_workers = 8
T = 2
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
store_images = True


class iCaRL_DM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        self.num_selection = args["num_selection"]
        self.is_dd = args["use_dd"]
        self.use_trajectory = args["use_trajectory"]
        self.selection = args["selection"]
        self.path = "logs/{}/{}/{}/{}/{}_{}_{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["selection"],
            args["num_selection"],
            args["seed"],
            args["convnet_type"],
        )
        self.datasets = args["dataset"]
        self._network = IncrementalNet(args, pretrained=True)
        self.dd = DistributionMatching(args)
        self.dsa_param = ParamDiffAug()
        self.dsa_strategy = dsa_strategy
        self.models = []
        self.increment = args['increment']
        self.inner_step = args['inner_step']
        self._real_data_memory, self._real_targets_memory = np.array(
            []), np.array([])

    def after_task(self):
        self._known_classes = self._total_classes

        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self.syn_loader = None
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes,
                                       self._total_classes)
        )
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )


        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # xxxxxxxxxxxxxxxxxxxxxxxx

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self._old_network = self._network.copy().freeze()
        self.build_rehearsal_memory(data_manager, self.samples_per_class, is_dd=self.is_dd, num_selection=self.num_selection)
        # self.inner_step = max(self.inner_step,math.ceil((self._cur_task+1)*self.increment/8))//2
        # self.inner_step = 3

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader,use_pretrained=True):
        self._network.to(self._device)

        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            if use_pretrained:
                with open('./ini_icarl_tiny_half', 'rb') as f:
                    self._network = pickle.load(f)
                    self._network.to(self._device)
                with open('./ini_icarl_tiny_half_models', 'rb') as f:
                    self.models = pickle.load(f)
                return
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            with open('./ini_icarl_tiny_half', 'wb') as f:
                pickle.dump(self._network.cpu(), f)
            with open('./ini_icarl_tiny_half_models', 'wb') as f:
                pickle.dump(self.models, f)
            self._network.to(self._device)
        else:
            # exmp_dataset = EpisodicTensorDataset(buffer_examplers, buffer_labels, ids_per_batch, ims_per_id)
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(
                train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(init_epoch))
        self._network.to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(
                    self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if self.use_trajectory:
                self.models.append(self._network.copy().freeze().to('cpu'))
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(epochs))
        syn_loader = self.get_random_loader(batch_size)
        syn_loader_iter = iter(syn_loader)
        for _, epoch in enumerate(prog_bar):
            # cnn_accy, nme_accy = self.eval_task()
            # print(cnn_accy["grouped"])
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                # inner roop with sync data and real data
                for j in range(self.inner_step):

                    # ckpoint1 = time.time()
                    # # inputs_syn, targets_syn = self.get_random_batch(
                    # #     batch_size)
                    # ckpoint2 = time.time()
                    # print('sample take time: ',ckpoint2-ckpoint1)
                    _, inputs_syn, targets_syn = next(syn_loader_iter)
                    # print(targets_syn)
                    # ckpoint3 = time.time()
                    # print('loader take time: ',ckpoint3-ckpoint2)
                    # inputs_syn = DiffAugment(inputs_syn, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    inputs_syn, targets_syn = inputs_syn.to(
                        self._device), targets_syn.to(self._device)
                    logits = self._network(inputs_syn)["logits"]

                    loss_clf = F.cross_entropy(logits, targets_syn)
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs_syn)["logits"],
                        T,
                    )

                    loss = (loss_clf+loss_kd)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets_syn.expand_as(preds)
                                        ).cpu().sum()
                    total += len(targets_syn)
                    # ckpoint4 = time.time()
                    # print('syn_com take time: ',ckpoint4-ckpoint3)
                # task data
                # ckpoint4 = time.time()

                inputs, targets = inputs.to(
                    self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = (loss_clf + loss_kd)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                # n step
                # self._network.convnet.dual_batch(1)
                self._network.to(self._device)
                # ckpoint5 = time.time()
                # print('orignal_com take time: ',ckpoint5-ckpoint4)
            # self._network.convnet.dual_batch(0)
            if self.use_trajectory:
                self.models.append(self._network.copy().freeze().to('cpu'))
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
        cnn_accy, nme_accy = self.eval_task()
        print(cnn_accy["grouped"])

    def _construct_exemplar_synthetic(self, data_manager, m, num_selection:int):
        logging.info(
            "Constructing exemplars for new classes...({} for old classes)".format(m)
        )
        classes_range = np.arange(self._known_classes, self._total_classes)
        data, targets, _ = data_manager.get_dataset(classes_range
            ,
            source="train",
            mode="test",
            ret_data=True,
        )
        # task3
        # theta2 = (D1+T2,theta1)
        # D1+D2+T3   D2= (theta2,ran), D1 = (theta1,T1)
        test_trsf = data_manager._test_trsf
        common_trsf = data_manager._common_trsf
        test_trsf = transforms.Compose([*test_trsf, *common_trsf])
        if data_manager.use_path:
            data = torch.stack([test_trsf(pil_loader(img)) for img in data]).numpy()
        else:
            data = torch.stack([test_trsf(img) for img in data]).numpy()
        # distill_data + task+data
        # real_data = np.concatenate((self._data_memory, data)) if len(self._data_memory) != 0 else data
        real_data = data
        # Select
        # real_label = np.concatenate((self._targets_memory, targets)) if len(self._targets_memory) != 0 else targets
        real_label = targets
        syn_data, syn_lablel = self.dd.gen_synthetic_data(
            m,self._old_network, self.models, real_data, real_label, classes_range, self.path, dataset_name=self.datasets,use_convents=False,use_trajectory=self.use_trajectory)
        if num_selection > 0:
            select_data, select_label = self.dd.select_sample(
                real_data, real_label, syn_data, syn_lablel.cpu(), self.models[-1], select_mode=self.selection,num_selection = num_selection)
            if self.datasets == 'cifar100':
                select_data = denormalize_cifar100(select_data)
            elif self.datasets == 'tinyimagenet200':
                select_data = denormalize_imageNet(select_data)
            else:
                select_data = denormalize_imageNet(select_data)
            select_data = tensor2img(select_data)
            if store_images:
                if data_manager.use_path:
                    select_data = save_images(select_data, select_label,
                                self.path, mode='select')
                else:
                    save_images(select_data, select_label,
                                self.path, mode='select')
            self._data_memory = (
                np.concatenate((self._data_memory, select_data))
                if len(self._data_memory) != 0
                else select_data
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, select_label))
                if len(self._targets_memory) != 0
                else select_label
            )
        if self.datasets == 'cifar100':
            syn_data = denormalize_cifar100(syn_data)
        elif self.datasets == 'tinyimagenet200':
            syn_data = denormalize_imageNet(syn_data)
        else:
            syn_data = denormalize_imageNet(syn_data)
        syn_data = tensor2img(syn_data)
        syn_lablel = syn_lablel.cpu().numpy()
        if store_images:
            if data_manager.use_path:
                syn_data = save_images(syn_data, syn_lablel, self.path, mode='sync')
            else:
                save_images(syn_data, syn_lablel, self.path, mode='sync')
        self._data_memory = (
            np.concatenate((self._data_memory, syn_data))
            if len(self._data_memory) != 0
            else syn_data
            )
        self._targets_memory = (
            np.concatenate((self._targets_memory, syn_lablel))
            if len(self._targets_memory) != 0
            else syn_lablel
            )

    def _construct_exemplar_random(self, data_manager, m):
        logging.info(
            "Selecting exemplars for new classes...({} for old classes)".format(
                m)
        )
        selected_exemplars = []
        exemplar_targets = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            num_ = len(data)
            inds = np.random.permutation(num_)[:m]
            selected_exemplars.extend(data[inds])
            exemplar_targets.extend(targets[inds])

        selected_exemplars = np.array(selected_exemplars)
        exemplar_targets = np.array(exemplar_targets)

        self._real_data_memory = (
            np.concatenate((self._real_data_memory, selected_exemplars))
            if len(self._real_data_memory) != 0
            else selected_exemplars
        )

        self._real_targets_memory = (
            np.concatenate((self._real_targets_memory, exemplar_targets))
            if len(self._real_targets_memory) != 0
            else exemplar_targets
        )

    def build_rehearsal_memory(self, data_manager, per_class, is_dd=True, num_selection=0):
        if self._fixed_memory:
            if is_dd:
                self._construct_exemplar_synthetic(
                    data_manager, per_class, num_selection)
            else:
                self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def get_random_batch(self, batch_size):
        """Returns a random batch according to current valid size."""
        # if global batch size > current valid size, we just sample with replacement
        replace = False if len(self._targets_memory) >= batch_size else True

        random_indices = np.random.choice(
            np.arange(len(self._targets_memory)), size=batch_size, replace=replace)

        image = self._data_memory[random_indices]
        label = self._targets_memory[random_indices]
        seed = int(time.time() * 1000) % 100000
        # image = DiffAugment(image, self.dsa_strategy, seed=seed, param=self.dsa_param)
        train_trsf = self.data_manager._train_trsf
        common_trsf = self.data_manager._common_trsf
        train_trsf = transforms.Compose([*train_trsf, *common_trsf])

        if self.data_manager.use_path:
            with Pool(8) as p:
                image = p.map(pil_loader,image) 
            image = torch.stack([train_trsf(img) for img in image])
        else:
            image = torch.stack([train_trsf(Image.fromarray(img)) for img in image])

        # data = [train_trsf(Image.fromarray(img)) for img in image]
        # image = torch.stack(data)
        return [image, torch.tensor(label)]
    def get_random_loader(self, batch_size):
        """Returns a random batch according to current valid size."""
        # if global batch size > current valid size, we just sample with replacement
        replace = False if len(self._targets_memory) >= batch_size else True
        train_trsf = self.data_manager._train_trsf
        common_trsf = self.data_manager._common_trsf
        trsf = transforms.Compose([*train_trsf, *common_trsf])
        syn_dataset = DummyDataset(self._data_memory, self._targets_memory, trsf, self.data_manager.use_path)
        repeat_sampler = RepeatSampler(len(syn_dataset), repeat=True,batch_size=batch_size)
        syn_dataloader = DataLoader(syn_dataset, batch_size=batch_size, sampler=repeat_sampler, num_workers=num_workers)

        # data = [train_trsf(Image.fromarray(img)) for img in image]
        # image = torch.stack(data)
        return syn_dataloader

    def get_random_real_batch(self, batch_size):
        """Returns a random batch according to current valid size."""
        global_bs = batch_size
        # if global batch size > current valid size, we just sample with replacement
        replace = False if len(
            self._real_targets_memory) >= global_bs else True

        random_indices = np.random.choice(
            np.arange(len(self._real_targets_memory)), size=global_bs, replace=replace)

        image = self._real_data_memory[random_indices]
        label = self._real_targets_memory[random_indices]
        # seed = int(time.time() * 1000) % 100000
        # image = DiffAugment(image, self.dsa_strategy, seed=seed, param=self.dsa_param)
        train_trsf = self.data_manager._train_trsf
        common_trsf = self.data_manager._common_trsf
        train_trsf = transforms.Compose([*train_trsf, *common_trsf])

        if self.data_manager.use_path:
            image = torch.stack([train_trsf(pil_loader(img)) for img in image])
        else:
            image = torch.stack([train_trsf(Image.fromarray(img)) for img in image])

        # data = [train_trsf(Image.fromarray(img)) for img in image]
        # image = torch.stack(data)
        return [image, torch.tensor(label)]


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
