import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args

        self.fc0 = nn.Linear(in_features=self.args.input_size, out_features=self.args.channels[0], bias=True)
        self.fc1 = nn.Linear(in_features=self.args.channels[0], out_features=self.args.channels[1], bias=True)
        self.fc2 = nn.Linear(in_features=self.args.channels[1], out_features=self.args.channels[2], bias=True)
        self.fc3 = nn.Linear(in_features=self.args.channels[2], out_features=self.args.channels[3], bias=True)
        self.fc4 = nn.Linear(in_features=self.args.channels[3], out_features=self.args.channels[4], bias=True)
        self.fc5 = nn.Linear(in_features=self.args.channels[4], out_features=2, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.fc0(x)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = act

        out = self.fc1(out)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = torch.cat((subnetwork, act), dim=1)

        out = self.fc2(out)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = torch.cat((subnetwork, act), dim=1)

        out = self.fc3(out)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = torch.cat((subnetwork, act), dim=1)

        out = self.fc4(out)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = torch.cat((subnetwork, act), dim=1)

        out = self.fc5(out)
        act = torch.where(out>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))
        out = self.act(out)
        subnetwork = torch.cat((subnetwork, act), dim=1)
    
        return out, subnetwork