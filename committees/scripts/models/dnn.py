import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self, input_size, output_size, width, depth, args):
        super(DNN, self).__init__()

        self.layers = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.layers.append(nn.Linear(input_size, width, bias=False))
            elif d == depth - 1:
                self.layers.append(nn.Linear(width, output_size, bias=False))
            else:
                self.layers.append(nn.Linear(width, width, bias=False))

    def init(self, type):
        if not hasattr(self, 'type'):
            self.type = type
        else:
            self.type = 'gaussian'

        for layer in self.layers:
            if self.type == 'gaussian':
                nn.init.normal_(layer[0].weight, mean=0.0, std=1.0)
            else:
                raise NotImplementedError

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x