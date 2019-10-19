import cv2
import glob
import sys
import torch

from loguru import logger as log
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """General dataset class allowing to load images from specified directory."""
    def __init__(self, dataset_directory, input_conversions_list, output_conversions_list, transform):
        self._dataset_directory = dataset_directory
        self._transform = transform
        self._input_conversions_list = input_conversions_list
        self._output_conversions_list = output_conversions_list

        self._file_types = [f'{self._dataset_directory}/*.jpg', f'{self._dataset_directory}/*.jpeg',
                            f'{self._dataset_directory}/*.png', f'{self._dataset_directory}/*.bmp']
        self._list_of_files_lists = [glob.glob(e) for e in self._file_types if glob.glob(e) != []]
        for l in self._list_of_files_lists:
            self._files_list = [img for img in l]

        log.info(f'List of loaded files: {self._files_list}')
        log.info(f'{len(self)} images loaded from: {self._dataset_directory}!')
        if not self._files_list:
            log.error(f'Images loading from {self._dataset_directory} failed!')
            sys.exit(1)

    def __len__(self):
        return len(self._files_list)

    @staticmethod
    def _implement_conversions(data, conversions_list):
        for conversion in conversions_list:
            data = conversion(data)
        return data


class BasicFiltersDataset(BaseDataset):
    """This class inherits from GeneralDataset to prepare converted and transformed inputs and outputs as
    connected samples. All operations are defined in "training_parameters.json" """
    def __init__(self, dataset_directory, input_conversions_list, output_conversions_list, transform=None):
        super().__init__(dataset_directory, input_conversions_list, output_conversions_list, transform)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_in = cv2.imread(self._files_list[item])
        image_out = image_in.copy()

        image_in = self._implement_conversions(image_in, self._input_conversions_list)
        image_out = self._implement_conversions(image_out,  self._output_conversions_list)

        sample = [image_in, image_out]

        if self._transform:
            sample = self._transform(sample)

        return sample
