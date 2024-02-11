import os
import numpy as np

import torch
from torchvision.datasets import FashionMNIST

def get_data():
    train_data = FashionMNIST(root='./data', train=True, download=False)
    test_data = FashionMNIST(root='./data', train=False, download=False)

    X_train = train_data.data.unsqueeze(1).float().flatten(start_dim=1)/255
    X_test = test_data.data.unsqueeze(1).float().flatten(start_dim=1)/255

    y_train = train_data.targets.long()
    y_test = test_data.targets.long()

    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)

    return train_data, test_data

if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.makedirs('./data')

    train_data = FashionMNIST(root='./data', train=True, download=True)
    test_data = FashionMNIST(root='./data', train=False, download=True)