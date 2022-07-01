import torch
import torch.nn as nn
import torch.nn.functional as F

class DGN(torch.nn.Module):
    def __init__(self, args):
        super(DGN, self).__init__()
        self.args = args
        
        self.fc0f = nn.ModuleList([nn.Linear(in_features=self.args.input_size, out_features=self.args.channels[0], bias=True) for _ in range(self.args.k)])
        self.fc0v = nn.Linear(in_features=self.args.input_size, out_features=self.args.channels[0], bias=True)

        self.fc1f = nn.ModuleList([nn.Linear(in_features=self.args.channels[0], out_features=self.args.channels[1], bias=True) for _ in range(self.args.k)])
        self.fc1v = nn.Linear(in_features=self.args.channels[0], out_features=self.args.channels[1], bias=True)
        
        self.fc2f = nn.ModuleList([nn.Linear(in_features=self.args.channels[1], out_features=self.args.channels[2], bias=True) for _ in range(self.args.k)])
        self.fc2v = nn.Linear(in_features=self.args.channels[1], out_features=self.args.channels[2], bias=True)

        self.fc3f = nn.ModuleList([nn.Linear(in_features=self.args.channels[2], out_features=self.args.channels[3], bias=True) for _ in range(self.args.k)])
        self.fc3v = nn.Linear(in_features=self.args.channels[2], out_features=self.args.channels[3], bias=True)

        self.fc4f = nn.ModuleList([nn.Linear(in_features=self.args.channels[3], out_features=self.args.channels[4], bias=True) for _ in range(self.args.k)])
        self.fc4v = nn.Linear(in_features=self.args.channels[3], out_features=self.args.channels[4], bias=True)

        self.fc5v = nn.Linear(in_features=self.args.channels[4], out_features=2, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        k_preact = [fc0f(x) for i, fc0f in enumerate(self.fc0f)]
        preact = gate([torch.where(pre>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device))  for pre in k_preact], self.args.exposure[0])
        subnetwork = preact
        out = self.fc0v(x) * preact

        k_preact = [fc1f(self.relu(k_preact[i])) for i, fc1f in enumerate(self.fc1f)]
        preact = gate([torch.where(pre>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device)) for pre in k_preact], self.args.exposure[1])
        subnetwork = torch.cat((subnetwork, preact), dim=1)
        out = self.fc1v(out) * preact

        k_preact = [fc2f(self.relu(k_preact[i])) for i, fc2f in enumerate(self.fc2f)]
        preact = gate([torch.where(pre>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device)) for pre in k_preact], self.args.exposure[2])
        subnetwork = torch.cat((subnetwork, preact), dim=1)
        out = self.fc2v(out) * preact

        k_preact = [fc3f(self.relu(k_preact[i])) for i, fc3f in enumerate(self.fc3f)]
        preact = gate([torch.where(pre>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device)) for pre in k_preact], self.args.exposure[3])
        subnetwork = torch.cat((subnetwork, preact), dim=1)
        out = self.fc3v(out) * preact

        k_preact = [fc4f(self.relu(k_preact[i])) for i, fc4f in enumerate(self.fc4f)]
        preact = gate([torch.where(pre>0, torch.ones(1,).to(self.args.device), torch.zeros(1,).to(self.args.device)) for pre in k_preact], self.args.exposure[4])
        subnetwork = torch.cat((subnetwork, preact), dim=1)
        out = self.fc4v(out) * preact

        out = self.fc5v(out)

        return out, subnetwork