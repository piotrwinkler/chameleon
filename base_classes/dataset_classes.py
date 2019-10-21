import cv2
import torch

from base_classes.data_collector import DataCollector
from torch.utils.data import Dataset
import numpy as np
from skimage import color
import pickle
from image_colorization.data.consts import choose_train_set, get_data_to_tests


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


class BasicCifar10Dataset(BaseDataset):
    L_rgb = []
    ab_rgb = []
    rgb_images = []
    L_mean = None
    L_std = None
    get_data_to_tests = None

    def __init__(self, dataset_directory, input_conversions_list, output_conversions_list, blur_details, transform=None):
        super().__init__(".", input_conversions_list, output_conversions_list, transform)

        self.get_data_to_tests = get_data_to_tests
        self.blur_details = blur_details
        train_set = choose_train_set

        if train_set == True:
            print("Loading train set")
            for i in range(1, 6):
                cifar_data_dict = self.unpickle(dataset_directory + "/data_batch_{}".format(i))
                if i == 1:
                    self.rgb_images = cifar_data_dict[b'data']
                else:
                    self.rgb_images = np.vstack((self.rgb_images, cifar_data_dict[b'data']))

        elif train_set == False:
            print("Loading test set")
            cifar_data_dict = self.unpickle(dataset_directory + "/test_batch")
            self.rgb_images = cifar_data_dict[b'data']

        self.rgb_images = self.rgb_images.reshape((len(self.rgb_images), 3, 32, 32))
        self.rgb_images = np.rollaxis(self.rgb_images, 1, 4)

        for img in self.rgb_images:
            lab_img = color.rgb2lab(img)
            self.L_rgb.append(lab_img[:, :, 0])
            self.ab_rgb.append(lab_img[:, :, 1:3])

        """
        After conversion to Lab, x set (L vector in Lab) is from 0 to 100
        After conversion to Lab, y set (ab vector in Lab) is from -128 to +127
        """

        self.L_rgb = np.array(self.L_rgb)
        self.ab_rgb = np.array(self.ab_rgb)

        if get_data_to_tests:
            self.L_mean = np.mean(self.L_rgb, axis=(0, 1, 2), keepdims=True)
            self.L_std = np.std(self.L_rgb, axis=(0, 1, 2), keepdims=True)

        self.L_rgb = self._implement_conversions(self.L_rgb, self._input_conversions_list)
        self.ab_rgb = self._implement_conversions(self.ab_rgb,  self._output_conversions_list)

        print("Dataset prepared")

    def __len__(self):
        return len(self.L_rgb)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.get_data_to_tests:

            curr_L = self.L_rgb[idx]
            if self.blur_details['do_blur']:
                curr_L = cv2.GaussianBlur(curr_L, tuple(self.blur_details['kernel_size']), 0)

            return curr_L[np.newaxis, :, :], np.transpose(self.ab_rgb[idx], (2, 0, 1))

        else:
            gray_img = color.rgb2gray(self.rgb_images[idx])
            gray_img = np.dstack((gray_img, gray_img, gray_img))
            L_gray = color.rgb2lab(gray_img)[:, :, 0]

            L_gray = self._implement_conversions(L_gray, self._input_conversions_list)

            if self.blur_details['do_blur']:
                L_gray_out = cv2.GaussianBlur(L_gray, tuple(self.blur_details['kernel_size']), 0)
            else:
                L_gray_out = L_gray

            return L_gray[np.newaxis, :, :], np.transpose(self.ab_rgb[idx], (2, 0, 1)), self.rgb_images[idx], \
                   L_gray_out[np.newaxis, :, :], gray_img

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as pickle_file:
            data = pickle.load(pickle_file, encoding='bytes')
        return data
