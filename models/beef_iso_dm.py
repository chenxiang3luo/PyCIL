import copy
import logging
import pickle
import time
from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import torch
from dd_algorithms.dm import DistributionMatching
from dd_algorithms.utils import (DiffAugment, ParamDiffAug, get_time,
                                 save_images)
from models.base import BaseLearner
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.data_manager import pil_loader
from utils.inc_net import BEEFISONet, CosineIncrementalNet, IncrementalNet
from utils.toolkit import (count_parameters, denormalize_cifar100,denormalize_imageNet,
                           target2onehot, tensor2img, tensor2numpy)
from utils.data_manager import DummyDataset,RepeatSampler

EPSILON = 1e-8
stor_images = True
batch_size = 128
class BEEFISO_DM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
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
        self._network = BEEFISONet(args, True)
        self.num_selection = args["num_selection"] 
        self.is_dd = args["use_dd"]
        self.use_trajectory = args["use_trajectory"] 
        self.selection = args["selection"]
        self._snet = None
        self.logits_alignment = args["logits_alignment"]
        self.val_loader = None
        self.reduce_batch_size = args["reduce_batch_size"]
        self.random = args.get("random",None)
        self.imbalance = args.get("imbalance",None)
        self.dd = DistributionMatching(args)
        self.models = []
        self.inner_step = args['inner_step']
    def after_task(self):
        self._network_module_ptr.update_fc_after()
        self._known_classes = self._total_classes
        if self.reduce_batch_size:
            if self._cur_task == 0: 
                self.args["batch_size"] = self.args["batch_size"]
            else:
                self.args["batch_size"] = self.args["batch_size"] * (self._cur_task+1) // (self._cur_task+2) 
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1 and self.args["is_compress"]:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc_before(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        print(self._network_module_ptr.old_fc)
        print(self._network.old_fc)
        if self._cur_task > 0:
            for id in range(self._cur_task):
                for p in self._network.convnets[id].parameters():
                    p.requires_grad = False
            for p in self._network.old_fc.parameters():
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
            pin_memory=True,
        )
        if self._cur_task > 0:
            if self.random or self.imbalance:
                val_dset = data_manager.get_finetune_dataset(known_classes=self._known_classes, total_classes=self._total_classes,
                                                         source="train", mode='train', appendent=self._get_memory(), type="ratio")
            else:
                _, val_dset = data_manager.get_dataset_with_split(np.arange(self._known_classes, self._total_classes),
                                                                       source='train', mode='train',
                                                                       appendent=self._get_memory(),
                                                                       val_samples_per_class=int(
                                                                           self.samples_old_class))
            self.val_loader = DataLoader(
                val_dset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader,self.val_loader)
        self._old_network = self._network.copy().freeze()
        if self.random or self.imbalance:
            self.build_rehearsal_memory_imbalance(data_manager,self.samples_per_class)
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class, is_dd=self.is_dd, num_selection = self.num_selection)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader, val_loader=None,use_pretrained=False):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            if use_pretrained:
                with open('./ini_beef_tiny_half', 'rb') as f:
                    self._network = pickle.load(f)
                    self._network.to(self._device)
                with open('./ini_beef_tiny_half_models', 'rb') as f:
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
            self.epochs = self.args["init_epochs"]
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            with open('./ini_beef_tiny_half', 'wb') as f:
                pickle.dump(self._network.cpu(), f)
            with open('./ini_beef_tiny_half_models', 'wb') as f:
                pickle.dump(self.models, f)
            self._network.to(self._device)
        else:

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["expansion_epochs"]
            )
            
            self.epochs = self.args["expansion_epochs"]
            self.state = "expansion"
            for p in self._network.biases.parameters():
                p.requires_grad = False
            self._expansion(train_loader, test_loader, optimizer, scheduler)
            
            
            
            for p in self._network_module_ptr.forward_prototypes.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.backward_prototypes.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.new_fc.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.convnets[-1].parameters():
                p.requires_grad = False
            for p in self._network.biases.parameters():
                p.requires_grad = True
            self.state = "fusion"
            self.epochs = self.args["fusion_epochs"]
            self.per_cls_weights = torch.ones(self._total_classes).to(self._device)
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=0.05,
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            for n, p in self._network.named_parameters():
                if p.requires_grad == True:
                    print(n)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["fusion_epochs"]
            )
            self._fusion(val_loader,test_loader,optimizer,scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_en = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                logits = self._network(inputs)["logits"]
                loss_en = self.args["energy_weight"] * self.get_energy_loss(inputs,targets,targets)
                loss = F.cross_entropy(logits, targets)
                loss = loss + loss_en
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_en += loss_en.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            if self.use_trajectory:
                self.models.append(self._network.copy().freeze().to('cpu'))
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch+1 % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        logging.info(info)

    def _expansion(self, train_loader, test_loader, optimizer, scheduler):
        if self.use_trajectory:
            self.models = []
        prog_bar = tqdm(range(self.epochs))
        syn_loader = self.get_random_loader(batch_size)
        syn_loader_iter = iter(syn_loader)
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_en = 0.0
            correct, total = 0, 0

            
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                for j in range(self.inner_step):
                    # inputs_syn, targets_syn = self.get_random_batch(
                    #         batch_size)
                    _, inputs_syn, targets_syn = next(syn_loader_iter)
                    inputs_syn, targets_syn = inputs_syn.to(
                        self._device, non_blocking=True
                    ), targets_syn.to(self._device, non_blocking=True)
                    outputs = self._network(inputs_syn)
                    logits,train_logits = (
                        outputs["logits"],
                        outputs["train_logits"]
                    )
                    pseudo_targets = targets_syn.clone()
                    for task_id in range(self._cur_task+1):
                        if task_id == 0:
                            pseudo_targets = torch.where(targets_syn<self.data_manager.get_accumulate_tasksize(task_id),task_id,pseudo_targets)
                        elif task_id == self._cur_task: 
                            pseudo_targets = torch.where(targets_syn-self._known_classes+1>0,targets_syn-self._known_classes+task_id,pseudo_targets)
                        else:
                            pseudo_targets = torch.where((targets_syn<self.data_manager.get_accumulate_tasksize(task_id)) & (targets_syn>self.data_manager.get_accumulate_tasksize(task_id-1)-1),task_id,pseudo_targets)
                    
                    train_logits[:, list(range(self._cur_task))] /= self.logits_alignment
                    loss_clf = F.cross_entropy(train_logits, pseudo_targets)
                    loss_fe = torch.tensor(0.).cuda(self.args["device"][0])
                    loss_en = self.args["energy_weight"]  * self.get_energy_loss(inputs_syn,targets_syn,pseudo_targets)
                    loss = loss_clf + loss_fe + loss_en
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    losses_fe += loss_fe.item()
                    losses_clf += loss_clf.item()
                    losses_en += loss_en.item()
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets_syn.expand_as(preds)).cpu().sum()
                    total += len(targets_syn)

                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits,train_logits = (
                    outputs["logits"],
                    outputs["train_logits"]
                )
                pseudo_targets = targets.clone()
                for task_id in range(self._cur_task+1):
                    if task_id == 0:
                        pseudo_targets = torch.where(targets<self.data_manager.get_accumulate_tasksize(task_id),task_id,pseudo_targets)
                    elif task_id == self._cur_task: 
                        pseudo_targets = torch.where(targets-self._known_classes+1>0,targets-self._known_classes+task_id,pseudo_targets)
                    else:
                        pseudo_targets = torch.where((targets<self.data_manager.get_accumulate_tasksize(task_id)) & (targets>self.data_manager.get_accumulate_tasksize(task_id-1)-1),task_id,pseudo_targets)
                
                train_logits[:, list(range(self._cur_task))] /= self.logits_alignment
                loss_clf = F.cross_entropy(train_logits, pseudo_targets)
                loss_fe = torch.tensor(0.).cuda(self.args["device"][0])
                loss_en = self.args["energy_weight"]  * self.get_energy_loss(inputs,targets,pseudo_targets)
                loss = loss_clf + loss_fe + loss_en
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_en += loss_en.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
            if self.use_trajectory:
                self.models.append(self._network.copy().freeze().to('cpu'))
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch+1 % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
            
    def _fusion(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            # self.
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                # for j in range(self.inner_step):
                #     inputs_syn, targets_syn = self.get_random_batch(
                #             batch_size)
                #     inputs_syn, targets_syn = inputs_syn.to(
                #         self._device, non_blocking=True
                #     ), targets_syn.to(self._device, non_blocking=True)
                #     outputs = self._network(inputs_syn)
                #     logits,train_logits = (
                #         outputs["logits"],
                #         outputs["train_logits"]
                #     )
                    
                #     loss_clf = F.cross_entropy(logits,targets_syn)                
                #     loss_fe = torch.tensor(0.).cuda(self.args["device"][0])
                #     loss_kd = torch.tensor(0.).cuda(self.args["device"][0])     
                #     loss = loss_clf + loss_fe + loss_kd
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                #     losses += loss.item()
                #     losses_fe += loss_fe.item()
                #     losses_clf += loss_clf.item()
                #     losses_kd += (
                #         self._known_classes / self._total_classes
                #     ) * loss_kd.item()
                #     _, preds = torch.max(logits, dim=1)
                #     correct += preds.eq(targets_syn.expand_as(preds)).cpu().sum()
                #     total += len(targets_syn)

                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits,train_logits = (
                    outputs["logits"],
                    outputs["train_logits"]
                )
                
                loss_clf = F.cross_entropy(logits,targets)                
                loss_fe = torch.tensor(0.).cuda(self.args["device"][0])
                loss_kd = torch.tensor(0.).cuda(self.args["device"][0])     
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
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
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch+1 % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
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
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


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


    def get_energy_loss(self,inputs,targets,pseudo_targets):
        inputs = self.sample_q(inputs)
        
        out = self._network(inputs)
        if self._cur_task == 0:
            targets = targets + self._total_classes
            train_logits, energy_logits = out["logits"], out["energy_logits"]
        else:
            targets = targets + (self._total_classes - self._known_classes) + self._cur_task
            train_logits, energy_logits = out["train_logits"], out["energy_logits"]
        
        logits = torch.cat([train_logits,energy_logits],dim=1)
        
        logits[:,pseudo_targets] = 1e-9        
        energy_loss = F.cross_entropy(logits,targets)
        return energy_loss

    def sample_q(self, replay_buffer, n_steps=3):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        self._network_copy = self._network_module_ptr.copy().freeze()
        init_sample = replay_buffer
        init_sample = torch.rot90(init_sample, 2, (2, 3))
        embedding_k = init_sample.clone().detach().requires_grad_(True)
        optimizer_gen = torch.optim.SGD(
            [embedding_k], lr=1e-2)
        for k in range(1, n_steps + 1):
            out = self._network_copy(embedding_k)
            if self._cur_task == 0:
                energy_logits, train_logits = out["energy_logits"], out["logits"]
            else:
                energy_logits, train_logits = out["energy_logits"], out["train_logits"]
            num_forwards = energy_logits.shape[1]
            logits = torch.cat([train_logits,energy_logits],dim=1)
            negative_energy = torch.log(torch.sum(torch.softmax(logits,dim=1)[:,-num_forwards:]))
            optimizer_gen.zero_grad()
            negative_energy.sum().backward()
            optimizer_gen.step()
            embedding_k.data += 1e-3 * \
                torch.randn_like(embedding_k)
        final_samples = embedding_k.detach()
        return final_samples
    
    
    def build_rehearsal_memory_imbalance(self, data_manager, per_class,is_dd=True):
        if self._fixed_memory:
            self._construct_exemplar_unified_imbalance(data_manager, per_class,self.random,self.imbalance)
        else:
            self._reduce_exemplar_imbalance(data_manager, per_class,self.random,self.imbalance)
            self._construct_exemplar_imbalance(data_manager, per_class,self.random,self.imbalance)
            
    def build_rehearsal_memory(self, data_manager, per_class,is_dd=True, num_selection=0):
        if self._fixed_memory:
            if is_dd:
                self._construct_exemplar_synthetic(data_manager, per_class, num_selection)
            else:
                self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _reduce_exemplar_imbalance(self, data_manager, m,random,imbalance):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            l = sum(mask)
            if l == 0:
                continue
            if random or imbalance is not None:
                dd, dt = dummy_data[mask][:-1], dummy_targets[mask][:-1]
            else:
                dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_imbalance(self, data_manager, m, random=False,imbalance=None):
        increment  = self._total_classes - self._known_classes

        if  random:
            '''
            uniform random type
            '''
            selected_exemplars = []
            selected_targets = []
            logging.info("Contructing exmplars, totally random...({} total instances  {} classes)".format(increment*m, increment))
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(self._known_classes,self._total_classes),source="train",mode="test",ret_data=True)
            selected_indices = np.random.choice(list(range(len(data))),m*increment,replace=False)
            for idx in selected_indices:
                selected_exemplars.append(data[idx])
                selected_targets.append(targets[idx])
            selected_exemplars = np.array(selected_exemplars)[:m*increment] 
            selected_targets = np.array(selected_targets)[:m*increment]
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if \
                    len(self._targets_memory) != 0 else selected_targets
        else:
            if imbalance is None:
                logging.info('Constructing exemplars...({} per classes)'.format(m))
                ms = np.ones(increment,dtype=int)*m
            elif imbalance>=1:
                '''
                half-half type
                '''
                ms=[m for _ in range(increment)]
                for i in range(increment//2):
                    ms[i]-=m//imbalance
                for i in range(increment//2,increment):
                    ms[i]+=m//imbalance
                np.random.shuffle(ms)
                ms = np.array(ms,dtype=int)
                logging.info("Constructing exmplars, Imbalance...({} or {} per classes)".format(m-m//imbalance,(m+m//imbalance)))
            elif imbalance<1: 
                '''
                exp type
                '''
                ms = np.array([imbalance**i for i in range(increment)])
                ms = ms/ms.sum()
                tot = m*increment
                ms = (tot*ms).astype(int)
                np.random.shuffle(ms)
                
            else:
                assert 0, "not implemented yet"
            logging.info("ms {}".format(ms))
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                      mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplar_vectors = []  # [n, feature_dim]
                for k in range(1, ms[class_idx-self._known_classes]+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

                # uniques = np.unique(selected_exemplars, axis=0)
                selected_exemplars = np.array(selected_exemplars)
                if len(selected_exemplars)==0:
                    continue
                exemplar_targets = np.full(ms[class_idx-self._known_classes], class_idx)
                self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
                self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                    len(self._targets_memory) != 0 else exemplar_targets

                # Exemplar mean
                idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                       appendent=(selected_exemplars, exemplar_targets))
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4,pin_memory=True)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                self._class_means[class_idx, :] = mean
                # self._class_means[class_idx, :] = class_mean

    def _construct_exemplar_unified_imbalance(self, data_manager, m,random,imbalance):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))
        increment  = self._total_classes - self._known_classes

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            if sum(mask) == 0: continue
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        if  random:
            '''
            uniform sample type
            '''
            selected_exemplars = []
            selected_targets = []
            logging.info("Contructing exmplars, totally random...({} total instances  {} classes)".format(increment*m, increment))
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(self._known_classes,self._total_classes),source="train",mode="test",ret_data=True)
            selected_indices = np.random.choice(list(range(len(data))),m*increment,replace=False)
            for idx in selected_indices:
                selected_exemplars.append(data[idx])
                selected_targets.append(targets[idx])
            selected_exemplars = np.array(selected_exemplars) 
            selected_targets = np.array(selected_targets)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if \
                    len(self._targets_memory) != 0 else selected_targets
        else:
            if imbalance is None:
                logging.info('Constructing exemplars...({} per classes)'.format(m))
                ms = np.ones(increment,dtype=int)*m
            elif imbalance>=1:
                '''
                half-half type
                '''
                ms=[m for _ in range(increment)]
                for i in range(increment//2):
                    ms[i]-=m//imbalance
                for i in range(increment//2,increment):
                    ms[i]+=m//imbalance
                np.random.shuffle(ms)
                ms = np.array(ms,dtype=int)
                logging.info("Constructing exmplars, Imbalance...({} or {} per classes)".format(m-m//imbalance,(m+m//imbalance)))
            elif imbalance<1: 
                '''
                exp type
                '''
                ms = np.array([imbalance**i for i in range(increment)])
                ms = ms/ms.sum()
                tot = m*increment
                ms = (tot*ms).astype(int)
                np.random.shuffle(ms)
                
            else:
                assert 0, "not implemented yet"
            logging.info("ms {}".format(ms))
            # Construct exemplars for new classes and calculate the means
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                     mode='test', ret_data=True)
                class_loader = DataLoader(class_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4,pin_memory=True)

                vectors, _ = self._extract_vectors(class_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplar_vectors = []
                for k in range(1, ms[class_idx-self._known_classes]+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

                selected_exemplars = np.array(selected_exemplars)
                if len(selected_exemplars)==0:
                    continue
                exemplar_targets = np.full(ms[class_idx-self._known_classes], class_idx)
                self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
                self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                    len(self._targets_memory) != 0 else exemplar_targets

                # Exemplar mean
                exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                         appendent=(selected_exemplars, exemplar_targets))
                exemplar_loader = DataLoader(exemplar_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(exemplar_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                _class_means[class_idx, :] = mean
                # _class_means[class_idx,:] = class_mean

            self._class_means = _class_means

    def _construct_exemplar_synthetic(self, data_manager, m, num_selection:int):
        logging.info(
            "Constructing exemplars for new classes...({} for old classes)".format(m)
        )
        classes_range = np.arange(self._known_classes, self._total_classes)
        data, targets, _ = data_manager.get_dataset(classes_range,
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
            m,self._old_network, self.models, real_data, real_label, classes_range, self.path,dataset_name = self.datasets,use_convents=True,use_trajectory=self.use_trajectory)
        if num_selection > 0:
            select_data, select_label = self.dd.select_sample(
                real_data, real_label, syn_data, syn_lablel.cpu(), self._old_network, select_mode=self.selection,num_selection = num_selection)
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

        # self._data_memory = (
        #     torch.cat((self._data_memory, syn_data))
        #     if len(self._data_memory) != 0
        #     else syn_data
        #     )
        # self._targets_memory = (
        #     torch.cat((self._targets_memory, syn_lablel))
        #     if len(self._targets_memory) != 0
        #     else syn_lablel
        #     )

        # self._data_memory = syn_data
        # self._targets_memory = syn_lablel

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
        syn_dataloader = DataLoader(syn_dataset, batch_size=batch_size, sampler=repeat_sampler, num_workers=8)

        # data = [train_trsf(Image.fromarray(img)) for img in image]
        # image = torch.stack(data)
        return syn_dataloader