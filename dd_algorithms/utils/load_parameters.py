import datetime
import os
import pickle
import sys
import time
from typing import Union
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from scipy import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import DataParallel

# from dd_algorithms.utils import may_make_dir


# def save_ckpt(modules_optims, ep, ckpt_file):
#     """
#     Save state_dict's of modules/optimizers to file.
#     Args:
#         modules_optims: A dict, {'mod':[list_of_modules],'opt':[list_of_opts]}
#         ep: the current epoch number
#         scores: the performance of current model
#         ckpt_file: The file path.
#     Note:
#         torch.save() reserves device type and id of tensors to save, so when
#         loading ckpt, you have to inform torch.load() to load these tensors to
#         cpu or your desired gpu, if you change devices.
#     """
#     mod_state_dicts = [m.state_dict() for m in modules_optims['mod']]
#     opt_state_dicts = [m.state_dict() for m in modules_optims['opt']]
#     ckpt = dict(mod_state_dicts=mod_state_dicts, opt_state_dicts=opt_state_dicts, ep=ep)
#     may_make_dir(os.path.dirname(os.path.abspath(ckpt_file)))
#     torch.save(ckpt, ckpt_file)


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
    """
    Load state_dict's of modules/optimizers from file.
    Args:
        modules_optims: A dict, {'mod':[list_of_modules],'opt':[list_of_opts]}
        ckpt_file: The file path.
        load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
            to cpu type.
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    # load model
    for m, sd in zip(modules_optims['mod'], ckpt['mod_state_dicts']):
        m.load_state_dict(sd)
    # load optimizer
    for m, sd in zip(modules_optims['opt'], ckpt['opt_state_dicts']):
        m.load_state_dict(sd)
    if verbose:
        print('Resume from ckpt {}, \nepoch {}'.format(ckpt_file, ckpt['ep']))
    return ckpt['ep']


def model_load_state_dict(model, src_state_dict):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.

    Arguments:
        model: A torch.nn.Module object.
        src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
        This is modified from torch.nn.modules.module.load_state_dict(), to make
        the warnings and errors more detailed.
    """
    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception as msg:
            print("Warning: Error occurs when copying '{}': {}"
                  .format(name, str(msg)))
    src_missing = set(dest_state_dict.keys()) - \
        set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        for n in src_missing:
            print('\t', n)
    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        for n in dest_missing:
            print('\t', n)

def get_parameters(model: Union[nn.Module, DataParallel], include='', not_include='', detach=True):
    # Extract the actual model from DataParallel wrapper if necessary
    # model = model.module if isinstance(model, DataParallel) else model
    # Helper function to process parameters based on conditions
    def process_parameter(p:torch.nn.Parameter):
        if detach:
            return p.detach().cpu()
        return p
    # Choose the right filter based on the include and not_include arguments
    if include:
        parameters = [process_parameter(p) for n, p in model.named_parameters() if include in n]
    elif not_include:
        parameters = [process_parameter(p) for n, p in model.named_parameters() if not_include not in n]
    else:
        parameters = [process_parameter(p) for p in model.parameters()]
    return parameters

def set_parameters(model:Union[nn.Module, DataParallel], parameters: list, include='', not_include=''):
    # if isinstance(model, nn.DataParallel):
    #     model = model.module
    if include == '' and not_include == '':
        for index, p in enumerate(model.parameters()):
            p.data.copy_(parameters[index])
    if include != '':
        for index, (n, p) in enumerate(model.named_parameters()):
            if include in n:
                p.data.copy_(parameters[index])
    if not_include != '':
        for index, (n, p) in enumerate(model.named_parameters()):
            if not_include not in n:
                p.data.copy_(parameters[index])

def set_zero_grad(model:Union[nn.Module, DataParallel]):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

def get_grad(model:Union[nn.Module, DataParallel], cpu=True):
    tmp_grad = []
    for p in model.parameters():
        if p.grad is not None:
            if cpu:
                tmp_grad.append(p.grad.detach().cpu())
            else:
                tmp_grad.append(p.grad.detach())
        else:
            if cpu:
                tmp_grad.append(torch.zeros_like(p).detach().cpu())
            else:
                device = p.device
                tmp_grad.append(torch.zeros_like(p).to(device))
    return tmp_grad