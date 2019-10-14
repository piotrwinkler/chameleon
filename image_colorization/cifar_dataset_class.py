from torch.utils.data import Dataset, DataLoader
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
    mean = None
    std = None

    def __init__(self, cifar_dir, train, preprocessing, do_blur, transform=None):

        print("Preparing dataset...")

        if train == True:
            for i in range(1, 6):
                cifar_data_dict = self.unpickle(cifar_dir + "/data_batch_{}".format(i))
                if i == 1:
                    cifar_data = cifar_data_dict[b'data']
                else:
                    cifar_data = np.vstack((cifar_data, cifar_data_dict[b'data']))

        elif train == False:
            cifar_data_dict = self.unpickle(cifar_dir + "/test_batch")
            cifar_data = cifar_data_dict[b'data']

        # cifar_data = self.unpickle(cifar_dir + "/data_batch_{}".format(1))[b'data']

        cifar_data = cifar_data.reshape((len(cifar_data), 3, 32, 32))
        cifar_data = np.rollaxis(cifar_data, 1, 4)

        # plt.imshow(self.cifar_data[0])
        # plt.show()

        # img = cv2.imread("datasets/pink.png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # img_Lab = color.rgb2lab(img)

        for img in cifar_data:
            lab_img = color.rgb2lab(img)
            self.x_data.append(lab_img[:, :, 0])
            self.y_data.append(lab_img[:, :, 1:3])

        """
        After conversion to Lab, x set (L vector in Lab) is from 0 to 100
        After conversion to Lab, y set (ab vector in Lab) is from -128 to +127
        """

        # self.x_train = [color.rgb2lab(cifar_img)[0] for cifar_img in cifar_data]
        # self.y_train = [color.rgb2lab(cifar_img)[1:3] for cifar_img in cifar_data]

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

        self.x_data = (self.x_data - 50) / 100

        if preprocessing == "standardization":
            # Standardization per channel
            print("Standardization on  ab channels")
            self.mean = np.mean(self.y_data, axis=(0, 1, 2), keepdims=True)
            self.std = np.std(self.y_data, axis=(0, 1, 2), keepdims=True)
            self.y_data = (self.y_data - self.mean) / self.std
        elif preprocessing == "normalization":
            print("Normalization on  ab channels")
            self.y_data = np.array(self.y_data) / 255

        # plt.imshow(self.x_data[0])
        # plt.show()

        if do_blur == True:
            print("Blurring L channel")
            for i in range(self.x_data.shape[0]):
                self.x_data[i] = cv2.GaussianBlur(self.x_data[i], (7, 7), 0)

        # self.x_data = cv2.GaussianBlur(self.x_data, (5, 5), 0)
        # blur = cv2.GaussianBlur(self.x_data[0], (5, 5), 0)

        # plt.imshow(blur)
        # plt.show()

        # plt.imshow(self.x_data[0])
        # plt.show()

        # plt.imshow(self.x_data[1])
        # plt.show()

        print("Dataset prepared")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # y = np.transpose(self.y_train[idx], (2, 0, 1))

        return self.x_data[idx], np.transpose(self.y_data[idx], (2, 0, 1))

    def unpickle(self, file):
        with open(file, 'rb') as pickle_file:
            data = pickle.load(pickle_file, encoding='bytes')
        return data
