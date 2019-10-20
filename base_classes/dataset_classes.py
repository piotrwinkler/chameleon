import cv2
import torch

from base_classes.data_collector import DataCollector
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """General dataset class allowing to load images from specified directory."""
    def __init__(self, dataset_directory, input_conversions_list, output_conversions_list, transform):
        self._dataset_directory = dataset_directory
        self._transform = transform
        self._input_conversions_list = input_conversions_list
        self._output_conversions_list = output_conversions_list

        self._files_list = DataCollector.collect_images(self._dataset_directory)

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

        # cv2.imshow(f'image_in', image_in)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # cv2.imshow(f'image_out', image_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        sample = [image_in, image_out]

        if self._transform:
            sample = self._transform(sample)

        return sample
