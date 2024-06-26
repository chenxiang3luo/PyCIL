import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def DiffAugment(x, strategy='', aug_methods=None):
    if strategy == 'None' or strategy == 'none':
        return x
    if strategy:
        assert isinstance(aug_methods, DiffAugMethod), 'Please init the aug_methods with DiffAugMethod class!'
        pbties = strategy.split('_')
        p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
        for f in aug_methods.AUGMENT_FNS[p]:
            x = f(x)
        x = x.contiguous()
    return x


class DiffAugMethod(object):
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.AUGMENT_FNS = {
            'color': [self.rand_brightness, self.rand_saturation, self.rand_contrast],
            'crop': [self.rand_crop],
            'cutout': [self.rand_cutout],
            'flip': [self.rand_flip],
            'scale': [self.rand_scale],
            'rotate': [self.rand_rotate],
        }


    # We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
    def rand_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale
        sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
    

    def rand_rotate(self, x): # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.ratio_rotate
        theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


    def rand_flip(self, x):
        prob = self.prob_flip
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        return torch.where(randf < prob, x.flip(3), x)


    def rand_brightness(self, x):
        ratio = self.brightness
        randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5)*ratio
        return x


    def rand_saturation(self, x):
        ratio = self.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x


    def rand_contrast(self, x):
        ratio = self.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x


    def rand_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x


    def rand_cutout(self, x):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x