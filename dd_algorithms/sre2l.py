import logging
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dd_algorithms.utils import *
from utils.toolkit import target2onehot, tensor2numpy
import copy
import time


# todo
Iteration = 4000
# Iteration = 1
ipc = 10
lr = 0.25
first_bn_multiplier = 10.
BN  =True
channel = 3 
tv_l2 = 0.000
l2_scale = 0.0000
im_size= [32,32]
batch_size = 64
r_bn = 0.01

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

class SRe2L():
    def __init__(self, args):

        self._device = args["device"][0]
    def gen_synthetic_data(self,old_model,initial_data,real_data,real_label,class_range):
        
        syn_img = []
        label_syn = []
        for ipc_id in range(ipc):
            print('ipc_id:',ipc_id)
            model_teacher = old_model.copy().freeze() # get a random model
            model_teacher.eval()
            num_class = len(class_range)
            best_cost = 1e4

            loss_r_feature_layers = []
            for module in model_teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_feature_layers.append(BNFeatureHook(module))
            targets_all = torch.LongTensor(class_range)
            for kk in range(0, num_class, batch_size):
                targets = targets_all[kk:min(kk+batch_size,num_class)].to(self._device)
                data_type = torch.float
                inputs = torch.randn((targets.shape[0], channel, im_size[0], im_size[1]), requires_grad=True, device=self._device,
                             dtype=data_type)
                optimizer = optim.Adam([inputs], lr=lr, betas=[0.5, 0.9], eps = 1e-8)
                lr_scheduler = lr_cosine_policy(lr, 0, Iteration)
                criterion = nn.CrossEntropyLoss()
                criterion = criterion.cuda(self._device)
                prog_bar = tqdm(range(Iteration))
                for iteration in prog_bar:
                    # learning rate scheduling
                    lr_scheduler(optimizer, iteration, iteration)

                    aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.4, 1),
                                            interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    ])
                    
                    inputs_jit = aug_function(inputs)

                    # forward pass
                    optimizer.zero_grad()
                    outputs = model_teacher(inputs_jit)['logits']

                    # R_cross classification loss
                    loss_ce = criterion(outputs, targets)

                    # R_feature loss
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                    loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                    # R_prior losses
                    _, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.reshape(batch_size, -1), dim=1).mean()

                    # combining losses
                    loss_aux = tv_l2 * loss_var_l2 + \
                            l2_scale * loss_l2 + \
                            r_bn * loss_r_bn_feature

                    loss = loss_ce + loss_aux

                    # do image update
                    loss.backward()
                    optimizer.step()

                    # clip color outlayers
                    inputs.data = clip(inputs.data)

                    if best_cost > loss.item() or iteration == 1:
                        best_inputs = inputs.data.clone()
                    info = "Loss {:.3f}".format(
                    loss
                )
                    prog_bar.set_description(info)
                syn_img.append(best_inputs)
                label_syn.append(targets)

                optimizer.state = collections.defaultdict(dict)

            torch.cuda.empty_cache()

        return [copy.deepcopy(torch.cat(syn_img).detach().cpu()),copy.deepcopy(torch.cat(label_syn).detach().cpu())]
    
