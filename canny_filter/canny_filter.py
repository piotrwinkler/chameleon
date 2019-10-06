import torch.nn as nn
import torch.nn.functional as f

from base_classes.filter_sketch import FilterSketch


class CannyFIlter(FilterSketch):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, padding=(1, 1))

    def forward(self, x):
        x = f.relu(self.conv1(x))
        return x
