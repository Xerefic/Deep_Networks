import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DLGN(nn.Module):

    def __init__(self, input_size, output_size, width, depth, args):
        super(DLGN, self).__init__()
        self.beta = args.beta

        self.npf = nn.ModuleList()
        self.npv = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.npf.append(nn.Linear(input_size, width, bias=False))
                self.npv.append(nn.Linear(input_size, width, bias=False))
            elif d == depth - 1:
                self.npf.append(nn.Linear(width, output_size, bias=False))
                self.npv.append(nn.Linear(width, output_size, bias=False))
            else:
                self.npf.append(nn.Linear(width, width, bias=False))
                self.npv.append(nn.Linear(width, width, bias=False))

        self.sigmoid = nn.Sigmoid()

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
        x_npf, x_npv = x, x
        for npf, npv in zip(self.npf, self.npv):
            x_npf = npf(x_npf)
            x_npv = npv(x_npv) * self.sigmoid(self.beta * x_npf)
        return x_npv