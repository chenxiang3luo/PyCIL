import copy
import logging
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional
from convs.conv_cifar import conv3
from convs.resnet import resnet18
from dd_algorithms.utils import (DiffAugment, ParamDiffAug, TensorDataset,
                                 get_time, model_load_state_dict,
                                 progressbar_tamplet, save_images,
                                 set_zero_grad,epoch,get_loops)
from models.base import BaseLearner
from models.icarl import iCaRL
from torch import nn, optim
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import (denormalize_cifar100, target2onehot, tensor2img,
                           tensor2numpy)

# todo
Iteration = 4000
# Iteration = 1
# ipc = 10
lr_img = 0.01
lr_net = 0.01
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
# dsa_strategy = None
BN  =True
stor_images = True
init = 'real'
channel = 3 
dsa = False if dsa_strategy in ['none', 'None'] else True
im_size= [32,32]
batch_real = 256
four_weight = [1.0,1.0,0.1,0.1]
inner_weight = 0.01
lambda_1 = 0.04
lambda_2 = 0.02

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    for milestone in [1200, 1600, 1800]:
        lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def criterion_middle(real_feature, syn_feature,class_range):
    MSE_Loss = nn.MSELoss(reduction='sum')
    shape_real = real_feature.shape
    real_feature = torch.mean(real_feature.view(len(class_range), shape_real[0] // len(class_range), *shape_real[1:]), dim=1)

    shape_syn = syn_feature.shape
    syn_feature = torch.mean(syn_feature.view(len(class_range), shape_syn[0] // len(class_range), *shape_syn[1:]), dim=1)

    return MSE_Loss(real_feature, syn_feature)
class CAFE():
    def __init__(self, args, pretrained = False):

        self._device = args["device"][0]
        self.dsa_param = ParamDiffAug()
        self.pretrained = pretrained
        self.args = args
    def gen_synthetic_data(self, m,old_model, models, real_data, real_label, class_range, save_path,use_convents:bool = False,use_trajectory:bool = False):
        # if self.pretrained:
        #     step = (total_class)//10
        #     images_train = torch.tensor(self.images_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
        #     labels_train = torch.tensor(self.labels_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
        #     print(labels_train)
        #     new_syn = [images_train,labels_train]
        #     return new_syn
        ipc = m
        outer_loop, inner_loop = get_loops(ipc)
        print(outer_loop, inner_loop)
        images_all = real_data
        labels_all = real_label
        indices_class = {c:[] for c in class_range}

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.tensor(images_all,dtype=torch.float).to(self._device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self._device)
        # total_class = class_range[-1]+1
        for c in class_range:
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c

            replace = False if len(indices_class[c]) >= n else True
            idx_shuffle = np.random.choice(indices_class[c], size=n, replace=replace)

            return images_all[idx_shuffle]
        # image_syn = torch.tensor(initial_data, dtype=torch.float, requires_grad=True, device=self._device)
        image_syn = torch.randn(size=(len(class_range)*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self._device)
        label_syn = torch.tensor(np.array([np.ones(ipc)*i for i in class_range]), dtype=torch.long, requires_grad=False, device=self._device).view(-1)
        if init == 'real':
            print('initialize synthetic data from random real images')
            for c in class_range:
                related_class = c-class_range[0]
                init_data = get_images(c, ipc).detach()
                
                if stor_images:
                    tmp_data = deepcopy(init_data)
                    tmp_data = denormalize_cifar100(tmp_data)
                    tmp_data = tensor2img(tmp_data)
                    tmp_label = np.array([c] * ipc)
                    save_images(tmp_data, tmp_label, save_path, mode = 'real')
                image_syn.data[related_class*ipc:(related_class+1)*ipc] = init_data.data
                # related_class = c-class_range[0]
                # image_syn.data[related_class*ipc:(related_class+1)*ipc] = get_images(c, ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self._device)
        criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(self._device)
        for it in tqdm(range(Iteration+1)):
            adjust_learning_rate(optimizer_img, it, lr_img)
            ''' Train synthetic data '''
            # net = conv3(pretrained=False,args=self.args) # get a random model
            if use_trajectory:
                net = deepcopy(random.choice(models[1:]))
            else:
                net = old_model.copy()
            if use_convents:
                net = net.convnets[-1]
            net = net.to(self._device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = True
            
            optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            self.dc_aug_param = None
            loss_avg = 0
            loss_kai = 0
            loss_middle_item = 0
            acc_watcher = list()
            pop_cnt = 0
            acc_test = 0.0

            while True:
            # for _ in range(1):
                ''' update synthetic data '''
                # if not BN: # for ConvNet
                #     loss = torch.tensor(0.0).to(self._device)
                #     for c in class_range:
                #         related_class = c-class_range[0]
                #         img_real = get_images(c, batch_real)
                #         img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))

                #         if dsa:
                #             seed = int(time.time() * 1000) % 100000
                #             img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
                #             img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)

                #         output_real = net(img_real).detach()
                #         output_syn = net(img_syn)

                #         loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                # else: # for BN
                images_real_all = []
                lab_real_all = []
                images_syn_all = []
                lab_syn_all = []
                loss = torch.tensor(0.0).to(self._device)
                for c in class_range:
                    related_class = c-class_range[0]
                    img_real = get_images(c, batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=self._device, dtype=torch.long) * related_class
                    img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((ipc,), device=self._device, dtype=torch.long) * c
                    if dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)
                    images_real_all.append(img_real)
                    lab_real_all.append(lab_real)
                    images_syn_all.append(img_syn)
                    lab_syn_all.append(lab_syn)
                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)
                lab_real_all = torch.cat(lab_real_all, dim=0)
                lab_syn_all = torch.cat(lab_syn_all, dim=0)

                output_real = net(images_real_all)
                output_syn = net(images_syn_all)
                real_logits = output_real['logits']
                
                real_features = output_real['fmaps']
                real_flatten = output_real['features']
                syn_features = output_syn['fmaps']
                syn_flatten = output_real['features']
                loss_middle = four_weight[-1] * criterion_middle(real_features[-1], syn_features[-1],class_range)  \
                                + four_weight[-2] * criterion_middle(real_features[-2], syn_features[-2],class_range) \
                                + four_weight[-3] * criterion_middle(real_features[-3], syn_features[-3],class_range) \
                                + four_weight[-4] * criterion_middle(real_features[-4], syn_features[-4],class_range)
                loss_real = criterion(real_logits[:,-len(class_range):], lab_real_all)
                loss += loss_middle
                loss += loss_real
                last_real_feature = torch.mean(real_flatten.view(len(class_range), int(real_flatten.shape[0] / len(class_range)), real_flatten.shape[1]), dim=1)
                last_syn_feature = torch.mean(syn_flatten.view(len(class_range), int(syn_flatten.shape[0] / len(class_range)), syn_flatten.shape[1]), dim=1)
                #2560*10 x 10 * n*10
                output = torch.mm(real_flatten, last_syn_feature.t())
                last_real_feature = torch.mean(
                    last_real_feature.unsqueeze(0).reshape(len(class_range), int(last_real_feature.shape[0] / len(class_range)),
                                                    last_real_feature.shape[1]), dim=1)
                print(output.shape)
                print(lab_real_all)
                loss_output = criterion_middle(last_syn_feature, last_real_feature,class_range) + inner_weight * criterion_sum(output[:,-len(class_range):], lab_real_all)
                loss += loss_output


                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                ############ for outloop testing ############
                for c in class_range:
                    img_real_test = get_images(c, 128)
                    lab_real_test = torch.ones((img_real_test.shape[0],), device=self._device, dtype=torch.long) * c
                    outputs = net(img_real_test)
                    prob = outputs['logits']
                    acc_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                acc_test /= len(class_range)
                acc_watcher.append(acc_test.detach().cpu())
                pop_cnt += 1
                # if len(acc_watcher) == 10:
                #     print('outer')
                #     print(max(acc_watcher) - min(acc_watcher))
                #     if max(acc_watcher) - min(acc_watcher) < lambda_1:
                #         acc_watcher = list()
                #         pop_cnt = 0
                #         acc_test = 0.0
                #         break
                #     else:
                #         acc_watcher.pop(0)
                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=128, shuffle=True,
                                                            num_workers=0)
                acc_inner_watcher = list()
                acc_syn_inner_watcher = list()
                pop_inner_cnt = 0
                acc_inner_test = 0
                count = 0
                while (1):
                # for _ in range(inner_loop):
                    count+=1
                    print(count)
                    inner_loss, inner_acc = epoch('train', trainloader, net, optimizer_net, criterion, dsa,dsa_strategy=dsa_strategy,dsa_param=self.dsa_param,device = self._device,
                                                aug=True if dsa else False)
                    acc_syn_inner_watcher.append(inner_acc)
                    for c in range(len(class_range)):
                        img_real_test = get_images(c, 128)
                        lab_real_test = torch.ones((img_real_test.shape[0],), device=self._device, dtype=torch.long) * c
                        outputs = net(img_real_test)
                        prob = outputs['logits']
                        acc_inner_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                    acc_inner_test /= len(class_range)
                    acc_inner_watcher.append(acc_inner_test.detach().cpu())
                    pop_inner_cnt += 1
                    if len(acc_inner_watcher) == 10:
                        print('inner')
                        print(max(acc_inner_watcher) - min(acc_inner_watcher))
                        print(max(acc_inner_watcher) - min(acc_inner_watcher) > lambda_2)
                        if max(acc_inner_watcher) - min(acc_inner_watcher) > lambda_2:
                            acc_inner_watcher = list()
                            acc_syn_inner_watcher = list()
                            pop_inner_cnt = 0
                            acc_inner_test = 0
                            break
                        else:
                            acc_inner_watcher.pop(0)
                    epoch('test', trainloader, net, optimizer_net, criterion, dsa,dsa_strategy=dsa_strategy,dsa_param=self.dsa_param,device = self._device,aug=True if dsa else False)
            if it%1000 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
        
        
        # if min_loss == -1.0 or min_loss >= loss_avg:
        #     min_loss = loss_avg
        new_syn = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
                

        logging.info("Exemplar size: {}".format(len(new_syn[0])))
        return new_syn

    def select_sample(self, real_data: np.ndarray =None, real_label: np.ndarray =None, sync_imgs: torch.Tensor =None, sync_labels: torch.Tensor =None, old_model: Union[nn.Module, DataParallel] =None,use_convents:bool=False, **kwargs):
        '''
        NOTE:
            Using the init_epoch to select the samples.
            Must set the backbone to the expert trajectory.
            Set the backbone to the init_epoch of expert trajectory.
        '''
        # select strategy
        select_mode = kwargs['select_mode']
        num_selection = kwargs['num_selection']
        assert real_data is not None and real_label is not None, 'The real data and real label can not be None!'
        real_label = torch.from_numpy(real_label).long().to(self._device)
        real_data = torch.from_numpy(real_data).float().to(self._device)

        tmp_examplers = defaultdict(list)

        if select_mode == 'random':
            print('-'*20+'Start Random Selecting Samples'+'-'*20)
            current_data = defaultdict(list)
            real_label = real_label.cpu().tolist()
            for index, tmp_label in enumerate(real_label):
                current_data[tmp_label].append(index)
            current_label = list(current_data.keys())
            real_data = real_data.data.cpu()
            for label in current_label:
                data_index = current_data[label]
                selected_index = random.sample(data_index, num_selection)
                tmp_examplers[label].append(real_data[selected_index])
            datas = []
            labels = []
            for key in tmp_examplers.keys():
                datas.append(torch.cat(tmp_examplers[key]))
                labels.extend([key]*num_selection)
            print(torch.cat(datas).shape,torch.tensor(labels).shape)
            return torch.cat(datas),torch.tensor(labels).numpy()
            # self._save_memory()
        elif select_mode == 'greedy':
            labels_syn = sync_labels.long()
            image_syn = sync_imgs.float()
            # save old status
            current_backbone_weight = deepcopy(old_model.state_dict())
            old_backbone_state = old_model.training

            net = deepcopy(old_model)
            if use_convents:
                net = net.convnets[-1]
            net = net.to(self._device)
            print('-'*20+'Start Greedy Selecting Samples'+'-'*20)

            # ----- set to init_epoch first
            # ----- later can choose from different start epoch from min_start_epoch to init_epoch

            set_zero_grad(net)

            # set data loader
            image_syn_eval, label_syn_eval = deepcopy(image_syn.detach()), deepcopy(labels_syn.detach())
            dst_train = TensorDataset(image_syn_eval, label_syn_eval)
            trainloader = DataLoader(dst_train, batch_size=batch_real, shuffle=True, num_workers=0)
            net.eval()
            
            with torch.no_grad():
                sync_features = []
                sync_label = []
                bar = progressbar_tamplet('Fetch SyncData Feature:', len(trainloader))
                for indx, datum in enumerate(trainloader):
                    img, label = datum[0], datum[1]
                    ims_var = img.float().to(self._device)
                    # if dsa_strategy is not None:
                    #     ims_var = DiffAugment(ims_var, strategy=dsa_strategy, param=self.dsa_param)
                    #     print(ims_var.shape)
                    output_sync = net(ims_var)['features'].detach()
                    sync_features.append(output_sync)
                    sync_label.append(label)
                    bar.update(indx+1)
                bar.finish()
            sync_features = torch.cat(sync_features, dim=0)
            sync_label = torch.cat(sync_label, dim=0)

            with torch.no_grad():
                real_features = []
                indices = torch.randperm(len(real_data))
                indices_chunks = list(torch.split(indices, batch_real))
                bar = progressbar_tamplet('Fetch RealData Feature:', len(indices_chunks))
                for i in range(len(indices_chunks)):
                    these_indices = indices_chunks.pop()
                    _real_imgs = real_data[these_indices]
                    # if dsa_strategy is not None:
                    #     _real_imgs = DiffAugment(_real_imgs, strategy=dsa_strategy, param=self.dsa_param)
                    output_real = net(_real_imgs)['features'].detach()
                    real_features.append(output_real)
                    bar.update(i+1)
                bar.finish()
            real_features = torch.cat(real_features, dim=0)
            
            # tmp_examplers = mp_model_greedy_select(ipc, net, deepcopy(sync_features), deepcopy(real_features), real_data, real_label)
            tmp_examplers = greedy_select(num_selection, deepcopy(sync_features), deepcopy(sync_label), deepcopy(real_features), real_data, real_label)
            model_load_state_dict(old_model, current_backbone_weight)
            old_model.train(old_backbone_state)

            datas = []
            labels = []
            for key in tmp_examplers.keys():
                datas.append(torch.stack(tmp_examplers[key]))
                labels.extend([key]*num_selection)
            print(torch.cat(datas).shape,torch.tensor(labels).shape)
            return torch.cat(datas),torch.tensor(labels).numpy()
    

def greedy_select(img_per_class, sync_features: torch.Tensor, sync_label: torch.Tensor, real_features: torch.Tensor, real_data: torch.Tensor, real_label: torch.Tensor):
    examplers = defaultdict(list)
    real_label = real_label.cpu().tolist()
    current_data = defaultdict(list)
    for index, tmp_label in enumerate(real_label):
        current_data[tmp_label].append(index)
    current_label = list(current_data.keys())
    class_num = len(current_label)
    # data_indexs = list(range(len(real_label)))
    # sync_mean_feature = torch.mean(sync_features, dim=0)
    # real_mean_feature = torch.mean(real_features, dim=0)
    # feature_num = len(sync_features)
    bar = progressbar_tamplet('Greedy Selecting:', class_num*img_per_class)
    for i_class, _class in enumerate(current_label):
        sync_mean_feature = torch.mean(sync_features[sync_label==_class], dim=0)
        real_mean_feature = torch.mean(real_features[real_label==_class], dim=0)
        data_indexs = deepcopy(current_data[_class])
        feature_num = len(sync_features[sync_label==_class])
        for img_num in range(img_per_class):
            min_dist = -1.0
            choosed_index = -1
            # score = []
            for index in data_indexs:
                selected = real_features[index]
                dist = torch.sum(((sync_mean_feature*feature_num + selected)/(feature_num+1) - real_mean_feature)**2).cpu().item()
                # score.append(dist)
                if min_dist == -1.0 or dist <= min_dist:
                    min_dist = dist
                    choosed_index = index
                    share_feature = deepcopy(selected)
            # print(score)
            # input()
            idx = choosed_index
            sync_mean_feature = (sync_mean_feature*feature_num + share_feature)/(feature_num+1)
            feature_num += 1
            tmp_label = real_label[idx]
            examplers[tmp_label].append(real_data[idx].data.cpu())
            data_indexs.remove(idx)
            bar.update(i_class*img_per_class+img_num+1)
    bar.finish()
    return examplers




def mp_model_greedy_select(img_per_class, net: Union[nn.Module, DataParallel], sync_features: torch.Tensor, real_features: torch.Tensor, real_data: torch.Tensor, real_label: torch.Tensor):
    real_label = real_label.cpu().tolist()
    current_data = defaultdict(list)
    for index, tmp_label in enumerate(real_label):
        current_data[tmp_label].append(index)
    current_label = list(current_data.keys())
    class_num = len(current_label)
    real_data = real_data.share_memory_()
    # init for multiprocess
    examplers = defaultdict(list)
    manager = mp.Manager()
    lock = manager.Lock()
    index_num = mp.Value('i', -1)
    finish_num = mp.Value('i', -1)
    min_dist = mp.Value('d', -1.0)
    choosed_index = mp.Value('i', -1)
    feature_num = mp.Value('i', len(sync_features))
    share_data_indexs = manager.list(range(len(real_label)))
    sync_mean_feature = torch.mean(sync_features, dim=0).share_memory_()
    real_mean_feature = torch.mean(real_features, dim=0).share_memory_()
    feature_share = torch.zeros_like(sync_mean_feature).to(sync_features.device)
    feature_share = feature_share.share_memory_()
    process_num = 10
    all_process = None
    event = manager.Event()
    event.clear()
    bar = progressbar_tamplet('Greedy Selecting:', class_num*img_per_class)
    for img_num in range(class_num*img_per_class):
        choosed_index.value = -1
        min_dist.value = -1.0
        index_num.value = -1
        finish_num.value = -1
        if all_process is None:
            all_process = []
            for _ in range(process_num):
                copy_net = deepcopy(net)
                copy_net.eval()
                set_zero_grad(copy_net)

                tmp_process = mp.Process(target=_model_greedy_select, args=(event, feature_num, index_num, finish_num, lock, share_data_indexs, real_data, sync_mean_feature, real_mean_feature, feature_share, copy_net, choosed_index, min_dist))
                all_process.append(tmp_process)
            for tmp_process in all_process:
                tmp_process.start()
        event.set()
        while True:
            # print(finish_num.value, '==', len(share_data_indexs)-1, 'index=', index_num.value)
            if finish_num.value == len(share_data_indexs)-1:
                break

        idx = choosed_index.value
        # print(name)
        tmp_mean_feature = (sync_mean_feature*feature_num.value + feature_share)/(feature_num.value+1)
        sync_mean_feature[:] = tmp_mean_feature
        feature_num.value += 1
        tmp_label = real_label[idx]
        examplers[tmp_label].append(real_data[idx].data.cpu())
        share_data_indexs.remove(idx)
        bar.update(img_num+1)
        if len(examplers[tmp_label]) == img_per_class:
            share_data_indexs[:] = list(set(share_data_indexs) - set(current_data[tmp_label]))
    if all_process is not None:
        for tmp_process in all_process:
            tmp_process.terminate()
        all_process = None
        event.clear()
    bar.finish()
    return examplers



def _model_greedy_select(event, feature_num, index_num, finish_num, lock, data_indexs, real_data, sync_mean_feature, real_mean_feature, share_feature, net, choosed_index, min_dist):
    while True:
        event.wait()
        with lock:
            index_num.value += 1
            current_index = index_num.value
            if current_index >= len(data_indexs):
                event.clear()
                # print('Data Done')
                continue
        index = data_indexs[current_index]
        data = real_data[index]
        with torch.no_grad():
            selected = net(data.unsqueeze(0).to(sync_mean_feature.device))['features'].detach()
        selected = selected.squeeze()
        dist = torch.sum(((sync_mean_feature*feature_num.value + selected)/(feature_num.value+1) - real_mean_feature)**2).cpu().item()
        # print("dist:", dist)
        with lock:
            if min_dist.value == -1.0 or dist < min_dist.value:
                min_dist.value = dist
                choosed_index.value = index
                share_feature[:] = deepcopy(selected)
            finish_num.value += 1