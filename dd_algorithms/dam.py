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
                                 set_zero_grad,get_attention)
from models.base import BaseLearner
from models.icarl import iCaRL
from torch import nn, optim
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import (denormalize_cifar100, denormalize_imageNet,target2onehot, tensor2img,
                           tensor2numpy)

# todo
Iteration = 20000
# Iteration = 1
# ipc = 10
lr_img = 0.01
lr_net = 0.01
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
BN  =True
stor_images = True
init = 'real'
channel = 3 
dsa = False if dsa_strategy in ['none', 'None'] else True
task_balance = 0.01
batch_real = 256

class DataDAM():
    def __init__(self, args, pretrained = False):

        self._device = args["device"][0]
        self.dsa_param = ParamDiffAug()
        self.pretrained = pretrained
        # if pretrained:
        #     fname = '/data2/chenxiang/PyCIL/res_DM_CIFAR100_ConvNetBN_10ipc.pt'           ``
        #     data = torch.load(fname, map_location='cpu')['data']
        #     self.images_train_all = data[0][0]
        #     self.labels_train_all = data[0][1]
        self.args = args
    def gen_synthetic_data(self, m,old_model, models, real_data, real_label, class_range, save_path,dataset_name:str = 'cifar100',use_convents:bool = False,use_trajectory:bool = False):
        # if self.pretrained:
        #     step = (class_range[-1]+1)//10
        #     images_train = torch.tensor(self.images_train_all[(step-1)*ipc*num_classes:step*ipc*num_classes])
        #     labels_train = torch.tensor(self.labels_train_all[(step-1)*ipc*num_classes:step*ipc*num_classes])
        #     print(labels_train)
        #     new_syn = [images_train,labels_train]
        #     return new_syn
        ipc = m
        num_classes = len(class_range)
        im_size= real_data.shape[-2:]
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
        image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self._device)
        label_syn = torch.tensor(np.array([np.ones(ipc)*i for i in class_range]), dtype=torch.long, requires_grad=False, device=self._device).view(-1)
        if init == 'real':
            print('initialize synthetic data from random real images')
            for c in class_range:
                related_class = c-class_range[0]
                init_data = get_images(c, ipc).detach()
                
                if stor_images:
                    tmp_data = deepcopy(init_data)
                    if dataset_name == 'cifar100':
                        tmp_data = denormalize_cifar100(tmp_data)
                    else:
                        tmp_data = denormalize_imageNet(tmp_data)
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
        print('%s training begins'%get_time())
        ''' Defining the Hook Function to collect Activations '''
        activations = {}
        def getActivation(name):
            def hook_func(m, inp, op):
                activations[name] = op.clone()
            return hook_func

        ''' Defining the Refresh Function to store Activations and reset Collection '''
        def refreshActivations(activations):
            model_set_activations = [] # Jagged Tensor Creation
            for i in activations.keys():
                model_set_activations.append(activations[i])
            activations = {}
            return activations, model_set_activations
        
        ''' Defining the Delete Hook Function to collect Remove Hooks '''
        def delete_hooks(hooks):
            for i in hooks:
                i.remove()
            return

        def attach_hooks(net):
            hooks = []
            for module in net.named_modules():
                if isinstance(module[1], nn.ReLU):
                    # Hook the Ouptus of a ReLU Layer
                    hooks.append(module[1].register_forward_hook(getActivation('ReLU_'+str(len(hooks)))))
            return hooks

        for it in tqdm(range(Iteration+1)):

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
                param.requires_grad = False
            

            loss_avg = 0.0
            min_loss = -1.0
            def error(real, syn, err_type="MSE"):
                        
                if(err_type == "MSE"):
                    err = torch.sum((torch.mean(real, dim=0) - torch.mean(syn, dim=0))**2)
                
                elif (err_type == "MAE"):
                    err = torch.sum(torch.abs(torch.mean(real, dim=0) - torch.mean(syn, dim=0)))
                    
                elif (err_type == "ANG"):
                    rl = torch.mean(real, dim=0) 
                    sy = torch.mean(syn, dim=0)
                    num = torch.matmul(rl, sy)
                    denom = (torch.sum(rl**2)**0.5) * (torch.sum(sy**2)**0.5)
                    err = torch.acos(num/denom)
                    
                elif(err_type == "MSE_B"):
                    err = torch.sum((torch.mean(real.reshape(num_classes, batch_real, -1), dim=1).cpu() - torch.mean(syn.cpu().reshape(num_classes, ipc, -1), dim=1))**2)
                elif(err_type == "MAE_B"):
                    err = torch.sum(torch.abs(torch.mean(real.reshape(num_classes, batch_real, -1), dim=1).cpu() - torch.mean(syn.reshape(num_classes, ipc, -1).cpu(), dim=1)))
                elif (err_type == "ANG_B"):
                    rl = torch.mean(real.reshape(F, batch_real, -1), dim=1).cpu()
                    sy = torch.mean(syn.reshape(num_classes, ipc, -1), dim=1)
                    
                    denom = (torch.sum(rl**2)**0.5).cpu() * (torch.sum(sy**2)**0.5).cpu()
                    num = rl.cpu() * sy.cpu()
                    err = torch.sum(torch.acos(num/denom))
                return err

            ''' update synthetic data '''
            loss = torch.tensor(0.0)
            mid_loss = 0
            out_loss = 0
            # else: # for BN
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

            hooks = attach_hooks(net)

            output_real = net(images_real_all)['features'].detach()
            activations, original_model_set_activations = refreshActivations(activations)



            output_syn = net(images_syn_all)['features']
            activations, syn_model_set_activations = refreshActivations(activations)
            delete_hooks(hooks)


            loss += torch.sum((torch.mean(output_real.reshape(num_classes, batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, ipc, -1), dim=1))**2)
            length_of_network = len(original_model_set_activations)

            for layer in range(length_of_network-1):
                
                real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')

                tl =  100*error(real_attention, syn_attention, err_type="MSE_B")
                loss+=tl
                mid_loss += tl

            output_loss =  100*task_balance * error(output_real, output_syn, err_type="MSE_B")

            loss += output_loss
            out_loss += output_loss

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            torch.cuda.empty_cache()

            loss_avg /= num_classes

            if it%1000 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
        
                if min_loss == -1.0 or min_loss >= loss_avg:
                    min_loss = loss_avg
                    new_syn = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
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
                # selected_index = random.sample(data_index, num_selection)
                selected_index = np.random.choice(data_index, size=num_selection, replace=False)
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
        np.random.shuffle(data_indexs)
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
                choosed_index.val