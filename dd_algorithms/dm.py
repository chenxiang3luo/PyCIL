import copy
import logging
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Union
from utils.toolkit import target2onehot, tensor2numpy,denormalize_cifar100,tensor2img
from dd_algorithms.utils import DiffAugment,ParamDiffAug,get_time,save_images
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional
from convs.conv_cifar import conv3
from convs.resnet import resnet18
from dd_algorithms.utils import DiffAugment, ParamDiffAug, get_time,model_load_state_dict
from models.base import BaseLearner
from models.icarl import iCaRL
from torch import nn, optim
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dd_algorithms.utils import (DiffAugMethod, TensorDataset, progressbar_tamplet,
                   set_zero_grad)
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
# todo
Iteration = 10000
# Iteration = 1
ipc = 10
lr_img = 0.01
lr_net = 0.01
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
BN  =True
use_trajectory = True
stor_images = True
init = 'real'
channel = 3 
dsa = False if dsa_strategy in ['none', 'None'] else True
im_size= [32,32]
batch_real = 256

class DistributionMatching():
    def __init__(self, args,pretrained = False):

        self._device = args["device"][0]
        self.dsa_param = ParamDiffAug()
        self.pretrained = pretrained
        if pretrained:
            fname = '/data2/chenxiang/PyCIL/res_DM_CIFAR100_ConvNetBN_10ipc.pt'
            data = torch.load(fname, map_location='cpu')['data']
            self.images_train_all = data[0][0]
            self.labels_train_all = data[0][1]
        self.args = args
    def gen_synthetic_data(self,old_model,models,real_data,real_label,class_range):
        if self.pretrained:
            step = (class_range[-1]+1)//10
            images_train = torch.tensor(self.images_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
            labels_train = torch.tensor(self.labels_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
            print(labels_train)
            new_syn = [images_train,labels_train]
            return new_syn
        images_all = real_data
        labels_all = real_label
        indices_class = {c:[] for c in class_range}

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.tensor(images_all,dtype=torch.float).to(self._device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self._device)

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
                    tmp_data = init_data
                    tmp_data = denormalize_cifar100(tmp_data)
                    tmp_data = tensor2img(tmp_data)
                    tmp_label = np.array([c] * ipc)
                    save_images(tmp_data, tmp_label,mode = 'real',arg = 'dm')
                image_syn.data[related_class*ipc:(related_class+1)*ipc] = init_data.data
        else:
            print('initialize synthetic data from random noise')
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        net = old_model.copy()
        net = net.to(self._device)
        for it in tqdm(range(Iteration+1)):

            ''' Train synthetic data '''
            # net = conv3(pretrained=False,args=self.args) # get a random model
            if use_trajectory:
                net = random.choice(models[1:])
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False


            loss_avg = 0

            ''' update synthetic data '''
            if not BN: # for ConvNet
                loss = torch.tensor(0.0).to(self._device)
                for c in class_range:
                    related_class = c-class_range[0]
                    img_real = get_images(c, batch_real)
                    img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))

                    if dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)

                    output_real = net(img_real).detach()
                    output_syn = net(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            else: # for BN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(self._device)
                for c in class_range:
                    related_class = c-class_range[0]
                    img_real = get_images(c, batch_real)
                    img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))

                    if dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)
 
                output_real = net(images_real_all)['features'].detach()
                output_syn = net(images_syn_all)['features']

                loss += torch.sum((torch.mean(output_real.reshape(len(class_range), batch_real, -1), dim=1) - torch.mean(output_syn.reshape(len(class_range), ipc, -1), dim=1))**2)



            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= len(class_range)

            if it%1000 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
        
        
            
        new_syn = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
                

        logging.info("Exemplar size: {}".format(len(new_syn[0])))
        return new_syn

    def select_sample(self, real_data: np.ndarray =None, real_label: np.ndarray =None, sync_imgs: torch.Tensor =None, sync_labels: torch.Tensor =None, old_model: Union[nn.Module, DataParallel] =None, **kwargs):
        '''
        NOTE:
            Using the init_epoch to select the samples.
            Must set the backbone to the expert trajectory.
            Set the backbone to the init_epoch of expert trajectory.
        '''
        # select strategy
        select_mode = kwargs['select_mode']
        assert real_data is not None and real_label is not None, 'The real data and real label can not be None!'
        real_label = torch.from_numpy(real_label).long().to(self._device)
        real_data = torch.from_numpy(real_data).float().to(self._device)

        tmp_examplers = defaultdict(list)

        if select_mode == 'random':
            current_data = defaultdict(list)
            real_label = real_label.cpu().tolist()
            for index, tmp_label in enumerate(real_label):
                current_data[tmp_label].append(index)
            current_label = list(current_data.keys())
            real_data = real_data.data.cpu()
            for label in current_label:
                data_index = current_data[label]
                selected_index = random.sample(data_index, ipc)
                tmp_examplers[label].append(real_data[selected_index])
            # self._save_memory()
        elif select_mode == 'greedy':
            labels_syn = sync_labels.long()
            image_syn = sync_imgs.float()
            # save old status
            current_backbone_weight = deepcopy(old_model.state_dict())
            old_backbone_state = old_model.training

            net = deepcopy(old_model)
            net = net.to(self._device)
            print('-'*20+'Start Selecting Samples'+'-'*20)

            # ----- set to init_epoch first
            # ----- later can choose from different start epoch from min_start_epoch to init_epoch

            set_zero_grad(net)

            # set data loader
            image_syn_eval, label_syn_eval = deepcopy(image_syn.detach()), deepcopy(labels_syn.detach())
            dst_train = TensorDataset(image_syn_eval, label_syn_eval)
            trainloader = DataLoader(dst_train, batch_size=batch_real, shuffle=True, num_workers=0)
            net.train()
            
            with torch.no_grad():
                sync_features = []
                bar = progressbar_tamplet('Fetch SyncData Feature:', len(trainloader))
                for indx, datum in enumerate(trainloader):
                    img = datum[0]
                    ims_var = img.float().to(self._device)
                    if dsa_strategy is not None:
                        ims_var = DiffAugment(ims_var, strategy=dsa_strategy, param=self.dsa_param)
                        print(ims_var.shape)
                    output_sync = net(ims_var)['features'].detach()
                    sync_features.append(output_sync)
                    bar.update(indx+1)
                bar.finish()
            sync_features = torch.cat(sync_features, dim=0)

            with torch.no_grad():
                real_features = []
                indices = torch.randperm(len(real_data))
                indices_chunks = list(torch.split(indices, batch_real))
                bar = progressbar_tamplet('Fetch RealData Feature:', len(indices_chunks))
                for i in range(len(indices_chunks)):
                    these_indices = indices_chunks.pop()
                    _real_imgs = real_data[these_indices]
                    if dsa_strategy is not None:
                        _real_imgs = DiffAugment(_real_imgs, strategy=dsa_strategy, param=self.dsa_param)
                    output_real = net(_real_imgs)['features'].detach()
                    real_features.append(output_real)
                    bar.update(i+1)
                bar.finish()
            real_features = torch.cat(real_features, dim=0)
            
            # tmp_examplers = mp_model_greedy_select(ipc, net, deepcopy(sync_features), deepcopy(real_features), real_data, real_label)
            tmp_examplers = greedy_select(ipc, net, deepcopy(sync_features), deepcopy(real_features), real_data, real_label)
            model_load_state_dict(old_model, current_backbone_weight)
            old_model.train(old_backbone_state)

        datas = []
        labels = []
        for key in tmp_examplers.keys():
            datas.append(torch.stack((tmp_examplers[key])))
            labels.extend([key]*ipc)
        print(torch.cat(datas).shape,torch.tensor(labels).shape)
        return torch.cat(datas),torch.tensor(labels).numpy()
    

def greedy_select(img_per_class, net: Union[nn.Module, DataParallel], sync_features: torch.Tensor, real_features: torch.Tensor, real_data: torch.Tensor, real_label: torch.Tensor):
    examplers = defaultdict(list)
    real_label = real_label.cpu().tolist()
    current_data = defaultdict(list)
    for index, tmp_label in enumerate(real_label):
        current_data[tmp_label].append(index)
    current_label = list(current_data.keys())
    class_num = len(current_label)
    data_indexs = list(range(len(real_label)))
    sync_mean_feature = torch.mean(sync_features, dim=0)
    real_mean_feature = torch.mean(real_features, dim=0)
    feature_num = len(sync_features)
    bar = progressbar_tamplet('Greedy Selecting:', class_num*img_per_class)
    for img_num in range(class_num*img_per_class):
        min_dist = -1.0
        choosed_index = -1
        for current_index in range(len(data_indexs)):
            index = data_indexs[current_index]
            selected = real_features[index]
            dist = torch.sum(((sync_mean_feature*feature_num + selected)/(feature_num+1) - real_mean_feature)**2).cpu().item()
            if min_dist == -1.0 or dist < min_dist:
                min_dist = dist
                choosed_index = index
                share_feature = deepcopy(selected)
        idx = choosed_index
        sync_mean_feature = (sync_mean_feature*feature_num + share_feature)/(feature_num+1)
        feature_num += 1
        tmp_label = real_label[idx]
        examplers[tmp_label].append(real_data[idx].data.cpu())
        data_indexs.remove(idx)
        bar.update(img_num+1)
        if len(examplers[tmp_label]) == img_per_class:
            data_indexs[:] = list(set(data_indexs) - set(current_data[tmp_label]))
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