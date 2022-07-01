from imports import *
from logic import *

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args

        self.conv0 = nn.Conv2d(in_channels=self.args.input_size, out_channels=self.args.channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=self.args.channels[0], out_channels=self.args.channels[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.args.channels[1], out_channels=self.args.channels[2], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=self.args.channels[2], out_channels=self.args.channels[3], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=self.args.channels[3], out_channels=self.args.channels[4], kernel_size=3, stride=1, padding=1, bias=True)
        self.fc5 = nn.Linear(in_features=self.args.channels[4], out_features=self.args.output_size, bias=True)

        self.gap = nn.AvgPool2d(kernel_size=32, stride=1, padding=0, ceil_mode=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv0(x)
        out = self.act(out)

        out = self.conv1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.act(out)

        out = self.gap(out).squeeze(-1).squeeze(-1)

        out = self.fc5(out)
    
        return out
