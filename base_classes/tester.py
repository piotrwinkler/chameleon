import cv2
import numpy as np
import torch

from base_classes.data_collector import DataCollector
from loguru import logger as log


class BaseTester:
    """Base class for all tests. When you create your own testing class you should inherit from this one, because it
    contains all parameters from "test_parameters.json" (Provided through entrypoint by SetupCreator) """
    def __init__(self, net_path, model, transforms, input_conversions_list, output_conversions_list, dataset_directory):
        self._dataset_directory = dataset_directory
        self._net_path = net_path
        self._model = model
        self._transforms = transforms
        self._input_conversions_list = input_conversions_list
        self._output_conversions_list = output_conversions_list

        self._files_list = DataCollector.collect_images(self._dataset_directory)

    def __len__(self):
        return len(self._files_list)

    @staticmethod
    def read_image(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def show_image(imgs_list):
        for i, img in enumerate(imgs_list):
            cv2.imshow(f'image{i}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _implement_transforms(data, transforms):
        for transform in transforms:
            data = transform(data)
        return data

    @staticmethod
    def _implement_conversions(data, conversions_list):
        for conversion in conversions_list:
            data = conversion(data)
        return data


class TestImgtoImg(BaseTester):
    """Class intended to perform tests of img to img networks."""
    def __init__(self, net_path, model, transforms, input_conversions_list, output_conversions_list, dataset_directory):
        super().__init__(net_path, model, transforms, input_conversions_list, output_conversions_list,
                         dataset_directory)

    def test(self):
        self._model.load_state_dict(torch.load(self._net_path))

        for i in range(len(self)):
            input_img = self.read_image(self._files_list[i])
            input_img = self._implement_conversions(input_img, self._input_conversions_list)
            orig_img = input_img.copy()
            input_img = self._implement_transforms([input_img], self._transforms).pop()
            input_img = input_img.unsqueeze(0)

            output_img = self._model(input_img)
            log.info(f'NN output: {output_img}')
            output_img = output_img.detach().numpy()
            output_img = np.squeeze(output_img)  # remove redundant dimensions
            output_img = output_img.transpose((1, 2, 0)) if len(np.shape(output_img)) == 3 else output_img
            output_img = self._implement_conversions(output_img, self._output_conversions_list)

            log.info(f'Original image shape: {np.shape(orig_img)}')
            log.info(f'Output image shape: {np.shape(output_img)}')

            self.show_image([orig_img, output_img])
