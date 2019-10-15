from torch.utils.data import Dataset
from torchvision import transforms, utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, color
import cv2


class CifarDataset(Dataset):

    x_data = []
    y_data = []
    ab_mean = None
    ab_std = None

    def __init__(self, cifar_dir, train, ab_preprocessing, L_processing, do_blur, kernel_size, transform=None):

        print("Preparing dataset...")

        if train == True:
            print("Loading train set")
            for i in range(1, 6):
                cifar_data_dict = self.unpickle(cifar_dir + "/data_batch_{}".format(i))
                if i == 1:
                    cifar_data = cifar_data_dict[b'data']
                else:
                    cifar_data = np.vstack((cifar_data, cifar_data_dict[b'data']))

        elif train == False:
            print("Loading test set")
            cifar_data_dict = self.unpickle(cifar_dir + "/test_batch")
            cifar_data = cifar_data_dict[b'data']

        # cifar_data = self.unpickle(cifar_dir + "/data_batch_{}".format(1))[b'data']

        cifar_data = cifar_data.reshape((len(cifar_data), 3, 32, 32))
        cifar_data = np.rollaxis(cifar_data, 1, 4)

        for img in cifar_data:
            lab_img = color.rgb2lab(img)
            self.x_data.append(lab_img[:, :, 0])
            self.y_data.append(lab_img[:, :, 1:3])

        """
        After conversion to Lab, x set (L vector in Lab) is from 0 to 100
        After conversion to Lab, y set (ab vector in Lab) is from -128 to +127
        """

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

        if L_processing == "normalization":
            print("Normalization on L channel")
            self.x_data = (self.x_data - 50) / 100

        # TODO: tests required
        elif L_processing == "standardization":
            print("Standardization on L channel")
            self.L_mean = np.mean(self.x_data, axis=(0, 1, 2), keepdims=True)
            self.L_std = np.std(self.x_data, axis=(0, 1, 2), keepdims=True)
            self.x_data = (self.x_data - self.L_mean) / self.L_mean

        if ab_preprocessing == "standardization":
            # Standardization per channel
            print("Standardization on ab channels")
            self.ab_mean = np.mean(self.y_data, axis=(0, 1, 2), keepdims=True)
            self.ab_std = np.std(self.y_data, axis=(0, 1, 2), keepdims=True)
            self.y_data = (self.y_data - self.ab_mean) / self.ab_std
        elif ab_preprocessing == "normalization":
            print("Normalization on ab channels")
            self.y_data = np.array(self.y_data) / 255

        if do_blur == True:
            print(f"Blurring L channel with kernel {kernel_size}")
            for i in range(self.x_data.shape[0]):
                self.x_data[i] = cv2.GaussianBlur(self.x_data[i], kernel_size, 0)

        print("Dataset prepared")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_data[idx][np.newaxis, :, :], np.transpose(self.y_data[idx], (2, 0, 1))

    def unpickle(self, file):
        with open(file, 'rb') as pickle_file:
            data = pickle.load(pickle_file, encoding='bytes')
        return data
