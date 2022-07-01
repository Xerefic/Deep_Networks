import numpy as np

import os
import copy
import pyprind
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from args import *
from utils import *
from data import *
from dnn import *
from dgn import *
from dlgn import *
from dlgnsf import *

class Trainer():
    def __init__(self, args, architecture):

        self.args = args

        self.traindata, self.validdata, self.testdata = self.args.data
        self.trainloader, self.validloader, self.testloader = self.get_iterator(self.args.data)
        
        self.model = self.get_model(architecture)
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        self.train_loss = []
        self.train_metrics = {'accuracy': [], 'subnetwork': {}}
        self.valid_loss = []
        self.valid_metrics = {'accuracy': [], 'subnetwork': {}}

        self.start_epoch = 0

    def get_iterator(self, data):
        train, valid, test = data
        trainloader = DataLoader(train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(self.args.seed))
        validloader = DataLoader(valid, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(self.args.seed))
        testloader = DataLoader(test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=np.random.seed(self.args.seed))
        return trainloader, validloader, testloader

    def get_criterion(self):
        return nn.CrossEntropyLoss(weight=self.args.weights).to(self.args.device)
    
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.max_epochs, eta_min=1e-12, last_epoch=-1, verbose=False)

    def get_model(self, architecture):
        model = architecture.to(self.args.device)
        return model

    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters())/1e6

    def save(self, epoch):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.args.checkpoint, "model.pth"))
        torch.save({
            'epoch': epoch,
            'args': args,
            'loss': (self.train_loss, self.valid_loss),
            'metrics': (self.train_metrics, self.valid_metrics)
            }, os.path.join(self.args.checkpoint, "metrics.pth"))
        
    def load(self):
        if os.path.exists(os.path.join(self.args.checkpoint, "model.pth")):
            checkpoints = torch.load(os.path.join(self.args.checkpoint, "model.pth"), map_location=self.args.device)
            self.model.load_state_dict(checkpoints['model_state_dict'])
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

        if os.path.exists(os.path.join(self.args.checkpoint, "metrics.pth")):
            checkpoints = torch.load(os.path.join(self.args.checkpoint, "metrics.pth"), map_location=self.args.device)
            self.args = checkpoints['args']
            self.train_loss, self.valid_loss = checkpoints['loss']
            self.train_metrics, self.valid_metrics = checkpoints['metrics']
            return checkpoints['epoch']
        return 0

    def train(self):
        epoch_loss = 0
        epoch_metrics = {'accuracy': 0, 'subnetwork': {}}

        torch.cuda.empty_cache()
        self.model.train()

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.trainloader), bar_char='█')
            for index, (data, label) in enumerate(self.trainloader):
                data = data.to(self.args.device).float()
                label = label.long().to(self.args.device)

                self.optimizer.zero_grad()
                
                output, subnetwork = self.model(data)

                loss = self.criterion(output, label)

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()/len(self.trainloader)

                epoch_metrics['accuracy'] += (torch.argmax(output, dim=1)==label).float().sum().item()/len(self.traindata)
                for item in range(data.size(0)):
                    id = ''.join([str(int(item)) for item in subnetwork[item, :].detach().cpu().tolist()])
                    if id in epoch_metrics['subnetwork'].keys():
                        epoch_metrics['subnetwork'][id].extend([data[item, :].detach().cpu().numpy()])
                    else:
                        epoch_metrics['subnetwork'][id] = [data[item, :].detach().cpu().numpy()]

                bar.update()
                torch.cuda.empty_cache()

        return epoch_loss, epoch_metrics

    def evaluate(self):
        epoch_loss = 0
        epoch_metrics = {'accuracy': 0, 'subnetwork': {}}

        torch.cuda.empty_cache()
        self.model.eval()

        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                bar = pyprind.ProgBar(len(self.validloader), bar_char='█')
                for index, (data, label) in enumerate(self.validloader):
                    data = data.to(self.args.device).float()
                    label = label.long().to(self.args.device)

                    output, subnetwork = self.model(data)

                    loss = self.criterion(output, label)

                    epoch_loss += loss.item()/len(self.validloader)
                    epoch_metrics['accuracy'] += (torch.argmax(output, dim=1)==label).float().sum().item()/len(self.validdata)
                    for item in range(data.size(0)):
                        id = ''.join([str(int(item)) for item in subnetwork[item, :].detach().cpu().tolist()])
                        if id in epoch_metrics['subnetwork'].keys():
                            epoch_metrics['subnetwork'][id].extend([data[item, :].detach().cpu().numpy()])
                        else:
                            epoch_metrics['subnetwork'][id] = [data[item, :].detach().cpu().numpy()]

                    bar.update()
                    torch.cuda.empty_cache()

        return epoch_loss, epoch_metrics

    def test(self):

        torch.cuda.empty_cache()
        self.model.eval()

        outputs = torch.empty([0,])

        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                bar = pyprind.ProgBar(len(self.testloader), bar_char='█')
                for index, (data, label) in enumerate(self.testloader):
                    data = data.to(self.args.device)

                    output, _ = torch.argmax(self.model(data)).detach().cpu()
                    outputs = torch.cat((outputs, output), dim=0)

                    bar.update()
                    torch.cuda.empty_cache()

        return outputs
    
    def fit(self, next=True):
        if next:
            self.start_epoch = self.load()

        for epoch in range(self.start_epoch+1, self.args.max_epochs+1, 1):

            epoch_train_loss, epoch_train_metrics = self.train()
            epoch_train_accuracy = epoch_train_metrics['accuracy']

            self.train_loss.append(epoch_train_loss)
            self.train_metrics['accuracy'].append(epoch_train_metrics['accuracy'])
            self.train_metrics['subnetwork'].append(epoch_train_metrics['subnetwork'])

            epoch_valid_loss, epoch_valid_metrics = self.evaluate()
            epoch_valid_accuracy = epoch_valid_metrics['accuracy']

            
            self.valid_loss.append(epoch_valid_loss)
            self.valid_metrics['accuracy'].append(epoch_valid_metrics['accuracy']) 
            self.valid_metrics['subnetwork'].append(epoch_valid_metrics['subnetwork']) 

            # self.scheduler.step()
            if epoch_valid_metrics['accuracy'] >= max(self.valid_metrics['accuracy']):
                self.save(epoch)

            time.sleep(1)
            print(f'Epoch {epoch}/{self.args.max_epochs} | Training: Loss = {round(epoch_train_loss, 4)}  Accuracy = {round(epoch_train_accuracy, 4)} | Validation: Loss = {round(epoch_valid_loss, 4)}  Accuracy = {round(epoch_valid_accuracy, 4)}')