from data import get_data
from models import *
from trainer import *
from helper import *

import os
import copy
import tqdm
import random
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FashionMNIST


seeds = [0, 50, 150, 300, 500, 1000, 1500, 2000, 2500, 3000]
lrs = [1e-2, 1e-3, 3e-4]
weight_decays = [1e-5, 5e-6, 1e-6]
betas = [1, 2, 4]
size_committees = [10]
num_neurons = [50, 100, 200, 500]
num_committees = np.arange(30, 100, 10).tolist() + np.arange(100, 1100, 100).tolist() + [2000, 3000, 4000, 5000]

class Args():
    seed = 0
    # device = 'cpu'
    device = torch. device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 784
    output_size = 10
    width = 10
    depth = 5

    num_neurons = None
    num_committees = None
    size_committee = None

    beta = None
    batch_size = 512
    lr = None
    weight_decay = None
    max_epochs = 200

def train_dnn(seeds, betas, lrs, weight_decays, logger):
    args_list = []
    for seed in seeds:
        for beta in betas:
            for lr in lrs:
                for weight_decay in weight_decays:
                    args = Args()
                    args.beta = beta
                    args.lr = lr
                    args.weight_decay = weight_decay
                    args.seed = seed
                    args_list.append(args)

    runs = []

    with Pool(min(os.cpu_count(), 15)) as p:
        runs += p.map(train_dnn_helper, args_list)

    return runs

def train_dlgn(seeds, betas, lrs, weight_decays, logger):
    args_list = []
    for seed in seeds:
        for beta in betas:
            for lr in lrs:
                for weight_decay in weight_decays:
                    args = Args()
                    args.beta = beta
                    args.lr = lr
                    args.weight_decay = weight_decay
                    args.seed = seed
                    args_list.append(args)

    runs = []

    with Pool(min(os.cpu_count(), 15)) as p:
        runs += p.map(train_dlgn_helper, args_list) 

    return runs   

def train_committees(seeds, betas, lrs, weight_decays, size_committees, num_neurons, num_committees, logger):
    args_list = []
    for seed in seeds:
        for beta in betas:
            for lr in lrs:
                for weight_decay in weight_decays:
                    for size_committee in size_committees:
                        for num_neuron in num_neurons:
                            for num_committee in num_committees:
                                args = Args()
                                args.beta = beta
                                args.lr = lr
                                args.weight_decay = weight_decay
                                args.seed = seed
                                args.num_neurons = num_neuron
                                args.num_committees = num_committee
                                args.size_committee = size_committee
                                args_list.append(args)

    runs = []

    with Pool(min(os.cpu_count(), 15)) as p:
        runs += p.map(train_committees_helper, args_list)

    return runs

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log_10.csv', mode='w')
    handler.setFormatter(formatter)
    if not len(logger.handlers): 
        logger.addHandler(handler)

    logger.info('model,beta,lr,weight_decay,seed,num_neurons,num_committees,size_committee,accuracy')
    train_networks = {
        'dnn': False,
        'dlgn': False,
        'committee': True
    }


    if train_networks['dnn']:
        runs = []
        logger.info('Training DNNs')
        runs = train_dnn(seeds, betas, lrs, weight_decays, logger)
        runs = pd.DataFrame(runs)
        runs.to_csv('dnn_10.csv', index=False)

    if train_networks['dlgn']:
        runs = []
        logger.info('Training DLGNs')
        runs = train_dlgn(seeds, betas, lrs, weight_decays, logger)
        runs = pd.DataFrame(runs)
        runs.to_csv('dlgn_10.csv', index=False)

    if train_networks['committee']:
        runs = []
        logger.info('Training Committees')
        runs = train_committees(seeds, betas, lrs, weight_decays, size_committees, num_neurons, num_committees, logger)
        runs = pd.DataFrame(runs)
        runs.to_csv('committee_10.csv', index=False)
    
    logger.info('Done')
