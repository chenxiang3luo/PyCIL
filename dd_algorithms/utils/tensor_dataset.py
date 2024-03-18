import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach().long()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    

class EpisodicTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, n_classes_per_step, n_samples_per_class):
        """
        data: n x c x h x w tensor or n index arange tensor
        labels: n tensor
        n_classes: Number of random classes to be included in each episode.
        n_samples: Number of samples per class in each episode.
        """
        self.data = data.detach().float()
        self.labels = labels.detach().long()
        self.data_idx = torch.arange(len(self.data))
        self.n_classes = n_classes_per_step
        self.n_samples = n_samples_per_class
        self.bs = self.n_classes*self.n_samples
        self.uniq_labels = torch.unique(self.labels)
        if len(self.uniq_labels)>1:
            self.uniq_labels = self.uniq_labels.tolist()
        else:
            self.uniq_labels = [self.uniq_labels.item()]

        # Organizing data by class
        self.data_idx_by_class = dict()
        self.data_trace = defaultdict(list)
        self.label_trace = defaultdict(list)
        for label in self.uniq_labels:
            self.data_idx_by_class[label] = self.data_idx[self.labels==label].tolist()

        self.make_task()

    def __len__(self):
        length = len(self.data_trace) 
        length = length if len(self.data_trace[length-1])!=0 else length-1
        return length

    def __getitem__(self, idx):
        return torch.cat(self.data_trace[idx]), torch.cat(self.label_trace[idx])
    

    def make_task(self, shuffle=True):
        for label in self.data_idx_by_class:
            if shuffle:
                random.shuffle(self.data_idx_by_class[label])
            num = len(self.data_idx_by_class[label])
            # to make sure every class can get n_samles
            while len(self.data_idx_by_class[label])%self.n_samples != 0:
                self.data_idx_by_class[label] += self.data_idx_by_class[label][:self.n_samples-num%self.n_samples]
        # make task trace
        self.data_trace.clear()
        self.label_trace.clear()
        labels = self.uniq_labels
        np.random.shuffle(labels)
        last_step_len = last_step_num = last_label = -1
        img_ptr = step_indx = 0
        while True:
            for _label in labels:
                idxs = self.data_idx_by_class[_label]
                if len(idxs) >= img_ptr+self.n_samples and last_label != _label:
                    self.data_trace[step_indx].append(self.data[idxs[img_ptr:img_ptr+self.n_samples]])
                    self.label_trace[step_indx].append(self.labels[idxs[img_ptr:img_ptr+self.n_samples]])
                    last_label = _label
                    if len(self.data_trace[step_indx])==self.n_classes:
                        step_indx += 1
            if last_step_num == len(self.data_trace) and last_step_len == len(self.data_trace[step_indx]):
                break
            last_step_num = len(self.data_trace)
            last_step_len = len(self.data_trace[step_indx])
            img_ptr += self.n_samples
