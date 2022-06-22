import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as poly

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import gzip

import os
import glob
import time
import random
import gc
import json
import copy
import pyprind
import tqdm
import itertools
import pickle as pkl
from dataclasses import dataclass, field
from typing import Union, List, Dict, Any, Optional, cast

import torch
import torchvision
import torchtext

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import args
import utils
import data
import dnn, dgn, dlgn, dlgnsf
import trainer


models = ['DNN', 'DGN', 'DLGN', 'DLGNSF']
seeds = [420, 999, 785, 2565, 3821, 856, 9999, 1001, 565, 7890]
mode = 'Same'
experiment = {}

for seed in seeds:
    for model in models:
        for k in range(1, 5+1, 1):
            exp = f'k{k}{model}_{mode}'
            name = f'k{k}{model}_{mode}_{seed}'

            args.seed = seed
            args.k = k
            args.exposure = exposures[args.k]

            trainer = get_trainer(args, model, mode)
            print(f'Model: {name}')
            _, valid_metrics = trainer.evaluate()

            subnetworks = len(valid_metrics['subnetwork'])
            get_plot(valid_metrics, name)
            print(f'Model: {name} | Subnetworks: {subnetworks}')

            if exp in experiment.keys():
                experiment[exp].append(subnetworks)
            else:
                experiment[exp] = [subnetworks]

with open('experiments.pkl', 'wb') as handle:
    pkl.dump(experiment, handle, protocol=pkl.HIGHEST_PROTOCOL)