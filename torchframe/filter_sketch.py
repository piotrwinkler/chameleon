"""Base filter sketch. Filter structure and forward method needs to be defined in python for all framework
implementations."""
import torch.nn as nn
from abc import ABC, abstractmethod


class FilterSketch(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        pass
