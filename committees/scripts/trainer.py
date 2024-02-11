import copy
import tqdm
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


class CommitteesTrainer():

    def __init__(self, data, models, optimzers, criterion, args):
        self.args = args
        
        self.traindata, self.testdata = data
        self.trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        self.npf, self.npv = models
        self.npf, self.npv = self.npf.to(self.args.device), self.npv.to(self.args.device)
        self.npf_opt, self.npv_opt = optimzers
        self.criterion = criterion

        self.loss = []
        self.accuracy = []

    def train_epoch(self, step):
        train_loss = 0
        train_accuracy = 0

        for idx, (x, y) in tqdm.tqdm(enumerate(self.trainloader)):
            self.npf.train()
            self.npv.train()
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            self.npf_opt.zero_grad()
            self.npv_opt.zero_grad()

            if step%5 != 0:
                z = self.npf(x)
                y_pred = self.npv(z)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.npf_opt.step()

            if step%5 == 0:
                z = self.npf(x)
                y_pred = self.npv(z)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.npv_opt.step()

            self.npf.eval()
            self.npv.eval()
            with torch.no_grad():
                z = self.npf(x)
                y_pred = self.npv(z)
                loss = self.criterion(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()

            train_loss += loss.item()/len(self.trainloader)
            train_accuracy += accuracy.item()/len(self.trainloader)

        return train_loss, train_accuracy
    
    def test(self):
        self.npf.eval()
        self.npv.eval()
        test_loss = 0
        test_accuracy = 0
        for idx, (x, y) in enumerate(self.testloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            with torch.no_grad():
                z = self.npf(x)
                y_pred = self.npv(z)
                loss = self.criterion(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()

            test_loss += loss.item()/len(self.testloader)
            test_accuracy += accuracy.item()/len(self.testloader)

        return test_loss, test_accuracy
    
    def train(self, epochs, verbose=True):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(epoch)
            self.loss.append(train_loss)
            self.accuracy.append(train_accuracy)

            test_loss, test_accuracy = self.test()

            if verbose:
                print(f'Epoch: {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}')
        
        return test_accuracy

class DNNTrainer():

    def __init__(self, data, model, optimzer, criterion, args):
        self.args = args
        
        self.traindata, self.testdata = data
        self.trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        self.model = model.to(self.args.device)
        self.optimizer = optimzer
        self.criterion = criterion

        self.loss = []
        self.accuracy = []

    def train_epoch(self):
        train_loss = 0
        train_accuracy = 0

        for idx, (x, y) in tqdm.tqdm(enumerate(self.trainloader)):
            self.model.train()
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()/len(self.trainloader)
            train_accuracy += (y_pred.argmax(dim=1) == y).float().mean().item()/len(self.trainloader)

        return train_loss, train_accuracy
    
    def test(self):
        self.model.eval()
        test_loss = 0
        test_accuracy = 0

        
        for idx, (x, y) in tqdm.tqdm(enumerate(self.testloader)):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

            test_loss += loss.item()/len(self.testloader)
            test_accuracy += (y_pred.argmax(dim=1) == y).float().mean().item()/len(self.testloader)

        return test_loss, test_accuracy
    
    def train(self, epochs, verbose=True):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch()
            self.loss.append([train_loss])
            self.accuracy.append([train_accuracy])

            test_loss, test_accuracy = self.test()

            if verbose:
                print(f'Epoch: {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}')
        
        return test_accuracy