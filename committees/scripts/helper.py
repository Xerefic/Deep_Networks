from data import get_data
from models import *
from trainer import *

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

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

global train_data, test_data
train_data, test_data = get_data()

global logger
logger = mp.get_logger()

def train_dnn_helper(args):
    torch.cuda.set_device(args.device)
    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    dnn = DNN(args.input_size, args.output_size, args.width, args.depth, args)
    dnn_opt = optim.SGD(dnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = DNNTrainer((train_data, test_data), dnn, dnn_opt, criterion, args)
    test_accuracy = trainer.train(args.max_epochs, verbose=False)

    run = {
        'model': 'DNN',
        'beta': args.beta,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'num_neurons': args.num_neurons,
        'num_committees': args.num_committees,
        'size_committee': args.size_committee,
        'accuracy': round(test_accuracy, 4)
    }

    log_message = f'DNN,{args.beta},{args.lr},{args.weight_decay},{args.seed},,,,{round(test_accuracy, 4)}'
    print(log_message)
    logger.info(log_message)
    logger.info(f'DNN,{args.beta},{args.lr},{args.weight_decay},{args.seed},,,,{round(test_accuracy, 4)}')
    
    return run

def train_dlgn_helper(args):
    torch.cuda.set_device(args.device)
    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    dlgn = DLGN(args.input_size, args.output_size, args.width, args.depth, args)
    dlgn_opt = optim.SGD(dlgn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = DNNTrainer((train_data, test_data), dlgn, dlgn_opt, criterion, args)
    test_accuracy = trainer.train(args.max_epochs, verbose=False)

    run = {
        'model': 'DLGN',
        'beta': args.beta,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'num_neurons': args.num_neurons,
        'num_committees': args.num_committees,
        'size_committee': args.size_committee,
        'accuracy': round(test_accuracy, 4)
    }

    log_message = f'DLGN,{args.beta},{args.lr},{args.weight_decay},{args.seed},,,,{round(test_accuracy, 4)}'
    print(log_message)
    logger.info(log_message)

    return run

def train_committees_helper(args):
    torch.cuda.set_device(args.device)
    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    npf = NPF(args.input_size, args.output_size, args.num_neurons, args.size_committee, args.num_committees, args)
    npv = NPV(args.input_size, args.output_size, args.num_neurons, args.size_committee, args.num_committees, args)
    
    npf_opt = optim.SGD(npf.parameters(), lr=args.lr, weight_decay=0)
    npv_opt = optim.SGD(npv.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = CommitteesTrainer((train_data, test_data), (npf, npv), (npf_opt, npv_opt), criterion, args)
    test_accuracy = trainer.train(args.max_epochs, verbose=False)

    run = {
        'model': 'Committee',
        'beta': args.beta,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'num_neurons': args.num_neurons,
        'num_committees': args.num_committees,
        'size_committee': args.size_committee,
        'accuracy': round(test_accuracy, 4)
    }

    log_message = f'Committee,{args.beta},{args.lr},{args.weight_decay},{args.seed},{args.num_neurons},{args.num_committees},{args.size_committee},{round(test_accuracy, 4)}'
    print(log_message)
    logger.info(log_message)

    return run