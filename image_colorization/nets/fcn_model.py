"""
Based on:
https://pdfs.semanticscholar.org/ca76/9bc02cb1b74a160d606fbb171afb13d0d615.pdf
"""

import torch.nn as nn
import torch.nn.functional as F


class FCN_net(nn.Module):
    def __init__(self):
        super(FCN_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))

        output = self.conv5(output)

        return output
