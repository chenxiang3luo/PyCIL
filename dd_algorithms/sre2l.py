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


class MyClass:
    pass
my = MyClass()
temperature = 2
use_fp16 = True
gradient_accumulation_steps = 2
my.mix_type = 'cutmix'
my.cutmix = 1.0
my.mode= 'fkd_save'
my.device = 'cuda:2'
fkd_batch = 20
fkd_epoch = 170
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
ini = 'no_real'
# sharing_strategy = "file_system"
# torch.multiprocessing.set_sharing_strategy(sharing_strategy)

# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)

class SRe2L():
    def __init__(self, args):

        self._device = args["device"][0]
    def gen_synthetic_data(self,old_model,initial_data,initial_label,real_data,real_label,class_range):
        label_to_images = {}
        for label, image in zip(initial_label,initial_data):
            if label in label_to_images:
                label_to_images[label].append(image)
            else:
                label_to_images[label] = [image]
        print(label_to_images.keys())
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
                if ini == 'real':
                    ini_data = []
                    for j in targets:
                        ini_data.append(label_to_images[j.item()][ipc_id])
                    ini_data = np.array(ini_data)
                    inputs = torch.tensor(ini_data,requires_grad=True, device=self._device,dtype=data_type)
                else:
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
    
    def gen_soft_label(self,model,syn_img,syn_label,num_exist_syn):
        soft_labels_epoches = []
        coords_epoches = []
        flip_epoches = []
        mix_indexs_epoches = [] 
        mix_lams_epoches = []
        mix_bboxs_epoches = []
        batch2img_epoches = []
        normalize = transforms.Normalize(mean = [0.5071, 0.4866, 0.4409],
        std = [0.2673, 0.2564, 0.2762])
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=32,
                                        scale=(0.08,
                                               1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = myDataset(syn_img, syn_label, transform)
        # generator = torch.Generator()
        # generator.manual_seed(1993)
        # sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=fkd_batch, shuffle=True,
        num_workers=4, pin_memory=True)
        model.eval()
        for epoch in tqdm(range(fkd_epoch)):
            coords = []
            flip = []
            mix_indexs = []
            mix_indexs = []
            mix_indexs = []
            mix_lams = []
            mix_bboxs = []
            soft_labels = []
            batch2imgs = []
            # data = [transform(Image.fromarray(img), None, None) for img in syn_img]
            # transposed = list(zip(*data))
            # sample_new, flip_status, coords_status = transposed[0],transposed[1],transposed[2]
            
            # sample_new = torch.stack(sample_new)
            # coords_status = torch.stack(coords_status)
            # flip_status = list(flip_status)
            # flip_status = torch.stack(flip_status)
            for batch_idx, (images, target, flip_status, coords_status,batch2img) in enumerate(train_loader):
                # print(images.shape) # [batch_size, 3, 224, 224]
                # print(flip_status.shape) # [batch_size,]
                # print(coords_status.shape) # [batch_size, 4]
                # print(batch2img)
                images = images.cuda(my.device)
                images, mix_index, mix_lam, mix_bbox = mix_aug(images,args=my)
                output = model(images)
                coords.append(coords_status.numpy())
                flip.append(flip_status.numpy())
                mix_indexs.append(mix_index.numpy())
                mix_lams.append(mix_lam)
                mix_bboxs.append(mix_bbox)
                soft_labels.append(output['logits'].cpu().numpy())
                batch2imgs.append(batch2img.numpy() + num_exist_syn)
            soft_labels_epoches.append(soft_labels)
            coords_epoches.append(coords)
            flip_epoches.append(flip)
            mix_indexs_epoches.append(mix_indexs)
            mix_lams_epoches.append(mix_lams)
            mix_bboxs_epoches .append(mix_bboxs)
            batch2img_epoches.append(batch2imgs)
        return coords_epoches,flip_epoches,mix_indexs_epoches,mix_lams_epoches,mix_bboxs_epoches,soft_labels_epoches,batch2img_epoches


class myDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.trsf is not None:
            sample_new, flip_status, coords_status = self.trsf(Image.fromarray(self.images[idx]), None, None)
        label = self.labels[idx]

        return sample_new, label, flip_status, coords_status,idx