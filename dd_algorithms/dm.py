import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from convs.conv_cifar import conv3
from convs.resnet import resnet18
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import copy
import time
from dd_algorithms.utils import DiffAugment,ParamDiffAug,get_time

# todo
Iteration = 10000
# Iteration = 1
ipc = 10
lr_img = 0.01
lr_net = 0.01
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
BN  =True
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
    def gen_synthetic_data(self,old_model,initial_data,real_data,real_label,class_range):
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
                image_syn.data[related_class*ipc:(related_class+1)*ipc] = get_images(c, ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        for it in tqdm(range(Iteration+1)):

            ''' Train synthetic data '''
            # net = conv3(pretrained=False,args=self.args) # get a random model
            net = old_model.copy()
            net = net.to(self._device)
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

