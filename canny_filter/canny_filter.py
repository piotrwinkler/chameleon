"""This class inherits from FilterSketch. It is crucial to fit NN structure with transforms and conversions implemented
during setup process and mentioned in according configuration .json files."""
import torch.nn as nn
import torch.nn.functional as f

from base_classes.filter_sketch import FilterSketch
from loguru import logger as log


class CannyFilter(FilterSketch):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2)

        log.info('NN structure defined!')

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))

        return x
