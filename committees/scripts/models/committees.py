import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NPF(nn.Module):
    
    def __init__(self, input_size, output_size, num_neurons, size_committee, num_committees, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_committees = num_committees

        self.neurons = nn.Parameter(torch.stack([torch.randn(output_size, input_size) for _ in range(num_neurons)])).to(args.device)
        self.npf = []
        for i in range(num_committees):
            while True:
                choice = torch.randint(0, num_neurons, size=(size_committee,))
                if len(torch.unique(choice)) == size_committee:
                    break
            self.npf.append(choice.tolist())
        self.npf = torch.tensor(self.npf).to(args.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = torch.sigmoid(self.neurons[self.npf] @ x.T)
        z = torch.prod(z, dim=1).permute(2, 0, 1)
        return z
    
class NPV(nn.Module):

    def __init__(self, input_size, output_size, num_neurons, size_committee, num_committees, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_committees = num_committees

        self.npv = nn.Linear(num_committees, 1, bias=False)

        self.init('gaussian')

    def init(self, type):
        if not hasattr(self, 'type'):
            self.type = type
        else:
            self.type = 'gaussian'

        if self.type == 'gaussian':
            nn.init.normal_(self.npv.weight, mean=0.0, std=1.0)
        else:
            raise NotImplementedError 

    def forward(self, z):
        z = self.npv(z.permute(0, 2, 1)) 
        return z.squeeze(-1)
