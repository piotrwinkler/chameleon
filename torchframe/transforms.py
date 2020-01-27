"""Transforms used by pytorch dataset and dataloader. Only ToTensor seems to be useful now.
Rest of the transforms may be removed in the future."""
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from loguru import logger as log


class Rescale:
    """Rescale the data in a sample to a given size."""

    def __init__(self, output_sizes_list):
        assert isinstance(output_sizes_list, list)
        self.output_sizes_list = output_sizes_list

    def __call__(self, sample):
        """The size for input and output data is applied separately."""
        assert isinstance(sample, list), log.error(f'Input sample {sample} is not a list in transform {self}')
        sample = [cv2.resize(data, tuple(output_size)) for data, output_size in zip(sample, self.output_sizes_list)]
        return sample


class Standardize:
    def __init__(self, standardization_factor=255):
        assert isinstance(standardization_factor, int)
        self.standardization_factor = standardization_factor

    def __call__(self, sample):
        assert isinstance(sample, list), log.error(f'Input sample {sample} is not a list in transform {self}')
        sample = [(data-np.mean(data)) / self.standardization_factor if np.mean(data) > 1
                  else data for data in sample]
        return sample


class Augment:
    def __init__(self):
        pass
        # TODO write augmentation logic


class RandomHorizontalFlip:
    def __init__(self):
        self.flip = transforms.Compose([transforms.RandomHorizontalFlip()])

    def __call__(self, array):
        return np.array(self.flip(Image.fromarray(array)))


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, data_type=torch.float):
        self.data_type = data_type

    def __call__(self, sample):
        """Input and output data are 2 or 3 channels images."""
        assert isinstance(sample, list), log.error(f'Input sample {sample} is not a list in transform {self}')
        # swap color axis because
        # numpy image: H x W x C
        # tensor image: C x H x W
        sample = [torch.from_numpy(np.expand_dims(data.transpose((0, 1)), axis=0)).type(self.data_type)
                  if len(np.shape(data)) == 2
                  else torch.from_numpy(data.transpose((2, 0, 1))).type(self.data_type) for data in sample]

        return sample
