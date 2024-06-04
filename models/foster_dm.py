import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.toolkit import target2onehot, tensor2numpy,denormalize_cifar100,denormalize_imageNet,tensor2img
from dd_algorithms.dm import DistributionMatching
from dd_algorithms.sre2l import SRe2L
from utils.data_manager import pil_loader
import time
from PIL import Image
import os
import pickle
from dd_algorithms.utils import DiffAugment,ParamDiffAug,get_time,save_images
EPSILON = 1e-8
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
# Please refer to https://github.com/G-U-N/ECCV22-FOSTER for the full source code to reproduce foster.
T = 2
EPSILON = 1e-8
batch_size = 128
stor_images = True

class FOSTER_DM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        self._network = FOSTERNet(args, True)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.is_dd = args["is_dd"]
        self.per_cls_weights = None
        self.num_selection = args["num_selection"] 
        self.use_trajectory = args["use_trajectory"] 
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.inner_step = 0
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()
        self.dd = DistributionMatching(args)
        self.selection = args["selection"]
        self.models = []
        self.path = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["selection"],
            args["seed"],
            args["convnet_type"],
        )
        self.datasets = args["dataset"]
    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            # is_dd = True
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self._old_network = self._network.copy().freeze()
        self.build_rehearsal_memory(data_manager, self.samples_per_class,is_dd=self.is_dd,num_selection = self.num_selection)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader,use_pretrained=True):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            if use_pretrained:
                with open('./ini_foster_tiny_half', 'rb') as f:
                    self._network = pickle.load(f)
                    self._network.to(self._device)
                with open('./ini_foster_tiny_half_models', 'rb') as f:
                    self.models = pickle.load(f)
                return
            
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            with open('./ini_foster_tiny_half', 'wb') as f:
                pickle.dump(self._network.cpu(), f)
            with open('./ini_foster_tiny_half_models', 'wb') as f:
                pickle.dump(self.models, f)
            self._network.to(self._device)
        else:

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            # cls_num_list_new = [
            #     self.samples_new_class(i)
            #     for i in range(self._known_classes, self._total_classes)
            # ]
            # cls_num_list = [self.inner_step*(np.sum(cls_num_list_new)/self._known_classes + self.samples_old_class)+ self.samples_old_class] * self._known_classes + cls_num_list_new

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"]
            )
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._feature_boosting(train_loader, test_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            # cls_num_list_new = [
            #     self.samples_new_class(i)
            #     for i in range(self._known_classes, self._total_classes)
            # ]
            # cls_num_list = [self.inner_step*(np.sum(cls_num_list_new)/self._known_classes + self.samples_old_class)+ self.samples_old_class] * self._known_classes + cls_num_list_new
            # effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list))
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
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
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for j in range(self.inner_step):
                    syn_inputs, syn_targets = self.get_random_batch(batch_size)
                    syn_inputs, syn_targets = syn_inputs.to(
                        self._device, non_blocking=True
                    ), syn_targets.to(self._device, non_blocking=True)
                    outputs = self._network(syn_inputs)
                    logits, fe_logits, old_logits = (
                        outputs["logits"],
                        outputs["fe_logits"],
                        outputs["old_logits"].detach(),
                    )
                    loss_clf = F.cross_entropy(logits / self.per_cls_weights, syn_targets)
                    loss_fe = F.cross_entropy(fe_logits, syn_targets)
                    loss_kd = self.lambda_okd * _KD_loss(
                        logits[:, : self._known_classes], old_logits, self.args["T"]
                    )
                    loss = loss_clf + loss_fe + loss_kd
                    optimizer.zero_grad()
                    loss.backward()
                    if self.oofc == "az":
                        for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                            if i == 0:
                                p.grad.data[
                                    self._known_classes :,
                                    : self._network_module_ptr.out_dim,
                                ] = torch.tensor(0.0)
                    elif self.oofc != "ft":
                        assert 0, "not implemented"
                    optimizer.step()
                    losses += loss.item()
                    losses_fe += loss_fe.item()
                    losses_clf += loss_clf.item()
                    losses_kd += (
                        self._known_classes / self._total_classes
                    ) * loss_kd.item()
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(syn_targets.expand_as(preds)).cpu().sum()
                    total += len(syn_targets)
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = (
                    outputs["logits"],
                    outputs["fe_logits"],
                    outputs["old_logits"].detach(),
                )
                loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits, targets)
                loss_kd = self.lambda_okd * _KD_loss(
                    logits[:, : self._known_classes], old_logits, self.args["T"]
                )
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            if self.use_trajectory:
                self.models.append(self._network.copy().freeze().to('cpu'))
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self.args, True)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"]
        )
        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for j in range(self.inner_step):
                    syn_inputs, syn_targets = self.get_random_batch(batch_size)
                    syn_inputs, syn_targets = syn_inputs.to(
                        self._device, non_blocking=True
                    ), syn_targets.to(self._device, non_blocking=True)
                    dark_logits = self._snet(syn_inputs)["logits"]
                    with torch.no_grad():
                        outputs = self._network(syn_inputs)
                        logits, old_logits, fe_logits = (
                            outputs["logits"],
                            outputs["old_logits"],
                            outputs["fe_logits"],
                        )
                    loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                    loss = loss_dark
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    _, preds = torch.max(dark_logits[: syn_targets.shape[0]], dim=1)
                    correct += preds.eq(syn_targets.expand_as(preds)).cpu().sum()
                    total += len(syn_targets)
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                dark_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class + self.num_selection
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
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
        # _, real_label,real_dataset = data_manager.get_dataset(
        #     classes_range,
        #     source="train",
        #     mode="test",
        #     ret_data=True,
        # )
        # distill_data + task+data
        # real_data = np.concatenate((self._data_memory, data)) if len(self._data_memory) != 0 else data
        real_data = data
        # Select
        # real_label = np.concatenate((self._targets_memory, targets)) if len(self._targets_memory) != 0 else targets
        real_label = targets
        syn_data, syn_lablel = self.dd.gen_synthetic_data(
            m,self._old_network, self.models, real_data, real_label, classes_range, self.path, dataset_name=self.datasets,use_convents=True,use_trajectory=self.use_trajectory)
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
            if stor_images:
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
        if stor_images:
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
        # syn_data, syn_lablel = self.dd.gen_synthetic_data(self._old_network,None,real_data,real_label,classes_range,None)
        # syn_data = denormalize_cifar100(syn_data)
        # syn_data = tensor2img(syn_data)
        # syn_lablel = syn_lablel.cpu().numpy()
        # # if stor_images:
        # #     save_images(syn_data, syn_lablel)
        # self._data_memory = (
        #     np.concatenate((self._data_memory, syn_data))
        #     if len(self._data_memory) != 0
        #     else syn_data
        #     )
        # self._targets_memory = (
        #     np.concatenate((self._targets_memory, syn_lablel))
        #     if len(self._targets_memory) != 0
        #     else syn_lablel
        #     )
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
        global_bs = batch_size
        # if global batch size > current valid size, we just sample with replacement
        replace = False if len(self._targets_memory) >= global_bs else True

        random_indices = np.random.choice(
            np.arange(len(self._targets_memory)), size=global_bs, replace=replace)

        image = self._data_memory[random_indices]
        label = self._targets_memory[random_indices]
        seed = int(time.time() * 1000) % 100000
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


