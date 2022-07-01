import numpy as np
import sklearn
import torch
from torchvision import transforms

from args import *
from utils import *

def get_data(size, range):
    def labelize(x):
        return torch.where(x[:,0]**2+x[:,1]**2>max(range[0], range[1])**2, torch.ones(1,), torch.zeros(1,))
    x = torch.linspace(range[0], range[1], steps=size)
    y = torch.linspace(range[0], range[1], steps=size)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.flatten().unsqueeze(-1)
    y = y.flatten().unsqueeze(-1)
    data = torch.cat((x, y), dim=1)
    label = labelize(data)
    return (data, label)

def get_weights(label):
    unique, counts = np.unique(label, return_counts=True)
    weights = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=unique, y=label))
    return weights

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        if self.mode == 'train':
            self.data = np.array(self.args.train_file[0])
            self.labels = np.array(self.args.train_file[1])
        elif self.mode == 'valid' or self.mode == 'test':
            self.data = np.array(self.args.test_file[0])
            self.labels = np.array(self.args.test_file[1])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)