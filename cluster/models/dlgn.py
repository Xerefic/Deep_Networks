from imports import *
from logic import *

class DLGN(torch.nn.Module):
    def __init__(self, args):
        super(DLGN, self).__init__()
        self.args = args
        
        self.conv0f = nn.ModuleList([nn.Conv2d(in_channels=self.args.input_size, out_channels=self.args.channels[0], kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.args.k)])
        self.conv0v = nn.Conv2d(in_channels=self.args.input_size, out_channels=self.args.channels[0], kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1f = nn.ModuleList([nn.Conv2d(in_channels=self.args.channels[0], out_channels=self.args.channels[1], kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.args.k)])
        self.conv1v = nn.Conv2d(in_channels=self.args.channels[0], out_channels=self.args.channels[1], kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv2f = nn.ModuleList([nn.Conv2d(in_channels=self.args.channels[1], out_channels=self.args.channels[2], kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.args.k)])
        self.conv2v = nn.Conv2d(in_channels=self.args.channels[1], out_channels=self.args.channels[2], kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3f = nn.ModuleList([nn.Conv2d(in_channels=self.args.channels[2], out_channels=self.args.channels[3], kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.args.k)])
        self.conv3v = nn.Conv2d(in_channels=self.args.channels[2], out_channels=self.args.channels[3], kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4f = nn.ModuleList([nn.Conv2d(in_channels=self.args.channels[3], out_channels=self.args.channels[4], kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.args.k)])
        self.conv4v = nn.Conv2d(in_channels=self.args.channels[3], out_channels=self.args.channels[4], kernel_size=3, stride=1, padding=1, bias=True)

        self.fc5v = nn.Linear(in_features=self.args.channels[4], out_features=self.args.output_size, bias=True)

        self.gap = nn.AvgPool2d(kernel_size=32, stride=1, padding=0, ceil_mode=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        k_preact = [conv0f(x) for i, conv0f in enumerate(self.conv0f)]
        preact = gate([self.sigmoid(self.args.beta*t) for t in k_preact], self.args.exposure[0], self.args.gating[0])
        out = self.conv0v(x) * preact

        k_preact = [conv1f(k_preact[i]) for i, conv1f in enumerate(self.conv1f)]
        preact = gate([self.sigmoid(self.args.beta*t) for t in k_preact], self.args.exposure[1], self.args.gating[1])
        out = self.conv1v(out) * preact

        k_preact = [conv2f(k_preact[i]) for i, conv2f in enumerate(self.conv2f)]
        preact = gate([self.sigmoid(self.args.beta*t) for t in k_preact], self.args.exposure[2], self.args.gating[2])
        out = self.conv2v(out) * preact

        k_preact = [conv3f(k_preact[i]) for i, conv3f in enumerate(self.conv3f)]
        preact = gate([self.sigmoid(self.args.beta*t) for t in k_preact], self.args.exposure[3], self.args.gating[3])
        out = self.conv3v(out) * preact

        k_preact = [conv4f(k_preact[i]) for i, conv4f in enumerate(self.conv4f)]
        preact = gate([self.sigmoid(self.args.beta*t) for t in k_preact], self.args.exposure[4], self.args.gating[4])
        out = self.conv4v(out) * preact

        out = self.gap(out).squeeze(-1).squeeze(-1)

        out = self.fc5v(out)

        return out
