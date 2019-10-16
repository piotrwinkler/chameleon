from torch.utils.data import Dataset
from torchvision import transforms, utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, color
import cv2


class CifarDataset(Dataset):

    L_rgb = []
    ab_rgb = []
    rgb_images = []
    ab_mean = None
    ab_std = None
    L_mean = None
    L_std = None
    get_data_to_tests = None
    L_processing = None
    do_blur = None
    kernel_size = None

    def __init__(self, cifar_dir, train, ab_preprocessing, L_processing, do_blur, kernel_size, get_data_to_tests, transform=None):

        print("Preparing dataset...")
        self.get_data_to_tests = get_data_to_tests
        self.L_processing = L_processing
        self.do_blur = do_blur
        self.kernel_size = kernel_size

        if train == True:
            print("Loading train set")
            for i in range(1, 6):
                cifar_data_dict = self.unpickle(cifar_dir + "/data_batch_{}".format(i))
                if i == 1:
                    self.rgb_images = cifar_data_dict[b'data']
                else:
                    self.rgb_images = np.vstack((self.rgb_images, cifar_data_dict[b'data']))

        elif train == False:
            print("Loading test set")
            cifar_data_dict = self.unpickle(cifar_dir + "/test_batch")
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

        if self.L_processing == "normalization":
            print("Normalization on L channel")
            self.L_rgb = (self.L_rgb - 50) / 100

        # TODO: tests required
        elif self.L_processing == "standardization":
            print("Standardization on L channel")
            L_mean = np.mean(self.L_rgb, axis=(0, 1, 2), keepdims=True)
            L_std = np.std(self.L_rgb, axis=(0, 1, 2), keepdims=True)
            self.L_rgb = (self.L_rgb - L_mean) / L_std

        if ab_preprocessing == "standardization":
            # Standardization per channel
            print("Standardization on ab channels")
            self.ab_mean = np.mean(self.ab_rgb, axis=(0, 1, 2), keepdims=True)
            self.ab_std = np.std(self.ab_rgb, axis=(0, 1, 2), keepdims=True)
            self.ab_rgb = (self.ab_rgb - self.ab_mean) / self.ab_std
        elif ab_preprocessing == "normalization":
            print("Normalization on ab channels")
            self.ab_rgb = np.array(self.ab_rgb) / 255

        if self.do_blur:
            print(f"Blurring L channel with kernel {self.kernel_size}")
            for i in range(self.L_rgb.shape[0]):
                self.L_rgb[i] = cv2.GaussianBlur(self.L_rgb[i], self.kernel_size, 0)

        print("Dataset prepared")

    def __len__(self):
        return len(self.L_rgb)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.get_data_to_tests:
            return self.L_rgb[idx][np.newaxis, :, :], np.transpose(self.ab_rgb[idx], (2, 0, 1))

        else:
            gray_img = color.rgb2gray(self.rgb_images[idx])
            gray_img = np.dstack((gray_img, gray_img, gray_img))
            L_gray = color.rgb2lab(gray_img)[:, :, 0]

            if self.L_processing == "normalization":
                print("Normalization on L_gray channel")
                L_gray = (L_gray - 50) / 100

            # TODO: tests required
            elif self.L_processing == "standardization":
                print("Standardization on L_gray channel")
                self.L_mean = np.mean(L_gray, axis=(0, 1, 2), keepdims=True)
                self.L_std = np.std(L_gray, axis=(0, 1, 2), keepdims=True)
                L_gray = (L_gray - self.L_mean) / self.L_std

            # if self.do_blur:
            #     print(f"Blurring L channel with kernel {self.kernel_size}")
            #     L_gray = cv2.GaussianBlur(L_gray, self.kernel_size, 0)

            return self.L_rgb[idx][np.newaxis, :, :], np.transpose(self.ab_rgb[idx], (2, 0, 1)), self.rgb_images[idx], \
                   L_gray[np.newaxis, :, :], gray_img

    def unpickle(self, file):
        with open(file, 'rb') as pickle_file:
            data = pickle.load(pickle_file, encoding='bytes')
        return data
