import cv2
import glob
import sys
import torch

from loguru import logger as log
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, dataset_directory, conversion_method, conversion_parameters, transform=None):
        self._dataset_directory = dataset_directory
        self._conversion_method = conversion_method
        self._conversion_parameters = conversion_parameters
        self._transform = transform
        self._files_list = glob.glob(f'{self._dataset_directory}/*.jpg')

        # TODO load many types of images
        # self._file_types = [f'{self._dataset_directory}/*.jpg'] #, f'{self._dataset_directory}/*.jpeg',
        # f'{self._dataset_directory}/*.png', f'{self._dataset_directory}/*.bmp']
        # self._list_of_files_lists = [glob.glob(e) for e in self._file_types if glob.glob(e) != []]
        # for l in self._list_of_files_lists:
        #    print(type(l))
        #    self._files_list.append(l)

        log.info(f'List of loaded files: {self._files_list}')
        log.info(f'{len(self)} images loaded from: {self._dataset_directory}!')
        if not self._files_list:
            log.error(f'Images loading from {self._dataset_directory} failed!')
            sys.exit(1)

    def __len__(self):
        return len(self._files_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_in = cv2.imread(self._files_list[item])
        image_out = self._conversion_method(image_in, *self._conversion_parameters)

        sample = {'image_in': image_in, 'image_out': image_out}

        if self._transform:
            sample = self._transform(sample)

        return sample
