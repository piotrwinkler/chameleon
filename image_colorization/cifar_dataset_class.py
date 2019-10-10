from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, color
import cv2


class CifarDataset(Dataset):

    x_train = []
    y_train = []

    def __init__(self, cifar_dir, transform=None):

        for i in range(1, 6):
            cifar_train_data_dict = self.unpickle(cifar_dir + "/data_batch_{}".format(i))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))

        # cifar_train_data = self.unpickle(cifar_dir + "/data_batch_{}".format(1))[b'data']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)

        # plt.imshow(self.cifar_train_data[0])
        # plt.show()

        # img = cv2.imread("datasets/pink.png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # img_Lab = color.rgb2lab(img)

        for img in cifar_train_data:
            self.x_train.append(color.rgb2lab(img)[:, :, 0])
            self.y_train.append(color.rgb2lab(img)[:, :, 1:3])
        # self.x_train = [color.rgb2lab(cifar_img)[0] for cifar_img in cifar_train_data]
        # self.y_train = [color.rgb2lab(cifar_img)[1:3] for cifar_img in cifar_train_data]

        # self.x_train = np.array(self.x_train)
        # self.y_train = np.array(self.y_train)

        self.x_train = (np.array(self.x_train) - 50) / 100
        self.y_train = np.array(self.y_train) / 255

        # cifar_test_data_dict = self.unpickle(cifar_dir + "/test_batch")
        # cifar_test_data = cifar_test_data_dict[b'data']
        #
        # cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
        # cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        #
        # for i in range(len(cifar_test_data)):
        #     self.x_test.append(color.rgb2lab(cifar_test_data[i])[0])
        #     self.y_test.append(color.rgb2lab(cifar_test_data[i])[1:3])
        #
        # self.x_test = np.array(self.x_test)
        # self.y_test = np.array(self.y_test)
        #
        # self.x_test = (self.x_test - 50) / 100
        # self.y_test = self.y_test / 255

        # plt.imshow(self.cifar_test_data[0])
        # plt.show()
        # print(len(self.cifar_train_data))

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # y = np.transpose(self.y_train[idx], (2, 0, 1))

        return self.x_train[idx], np.transpose(self.y_train[idx], (2, 0, 1))

    def unpickle(self, file):
        with open(file, 'rb') as pickle_file:
            data = pickle.load(pickle_file, encoding='bytes')
        return data
