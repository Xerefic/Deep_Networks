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


@dataclass
class TrainingArgs():

    seed: int = 17
    lr: float = 3e-4
    batch_size: int = 4096
    num_workers: int = os.cpu_count()
    max_epochs: str = 100
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size: int = 2
    channels: tuple = (16, 16, 16, 16, 16)
    k: int = 2
    beta: int = 1
    exposure: tuple = tuple(torch.where(torch.rand(len(channels), 2**k)>0.5, torch.ones(1,), torch.zeros(1,)).tolist())

    train_file: tuple = None
    valid_file: tuple = None
    test_file: tuple = None
    data: tuple = None
    checkpoint: str = './'

    project_name: str = 'DLGN - NN'

args = TrainingArgs()

