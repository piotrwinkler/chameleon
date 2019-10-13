# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Check how YUV were normalized in paper
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net1
import torch
import torch.nn as nn
import time
import torch.optim as optim
from base_classes.logger_class import Logger
import sys
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from skimage import io, color
from image_colorization.cifar_dataset_class import CifarDataset


dataset_path = 'datasets/Cifar-10/cifar-10-batches-py'

which_version = "V7"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"

batch_size = 1


def main():

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    cifar_dataset = CifarDataset(dataset_path)
    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    net = FCN_net1()
    net = net.double()
    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    net.load_state_dict(torch.load(load_net_file))
    net.eval()

    criterion = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data

            img_rgb = np.transpose(inputs[0].numpy(), (1, 2, 0))
            # plt.imshow(img_rgb)
            # plt.show()

            # lab = color.rgb2lab(img_rgb)

            Y_batch, ab_batch = yuv_convert(inputs)

            y_original = np.transpose(Y_batch[0].numpy(), (1, 2, 0))
            y_original = y_original*100 + 50
            ab_original = np.transpose(ab_batch[0].numpy(), (1, 2, 0))
            ab_original = ab_original * 255

            img_rgb_original = color.lab2rgb(np.dstack((y_original, ab_original)))
            # plt.imshow(img_rgb_original)
            # plt.show()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(img_rgb_original)
            gray = color.rgb2gray(img_rgb_original)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(gray, cmap=plt.get_cmap('gray'))
            # zero the parameter gradients
            # forward + backward + optimize
            outputs = net(Y_batch)
            loss = criterion(outputs, ab_batch)

            ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
            ab_outputs = ab_outputs * 255

            img_rgb_outputs = color.lab2rgb(np.dstack((y_original, ab_outputs)))
            # plt.imshow(img_rgb_outputs)
            # plt.show()
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(img_rgb_outputs)

            plt.show()

            running_loss = loss.item()
            # if i % 5 == 4:  # print every 5 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, (i + 1) * batch_size, running_loss / 5))
            #     running_loss = 0.0

            print(f'[{(i + 1) * batch_size}] loss: {running_loss}')
            # break

    print('Finished Testing')


def yuv_convert(imgs_batch):
    Y_batch = []
    ab_batch = []

    for i in range(imgs_batch.shape[0]):
        img_rgb = np.transpose(imgs_batch[i].numpy(), (1, 2, 0))
        # plt.imshow(img_rgb)
        # plt.show()
        img_Lab = color.rgb2lab(img_rgb)

        # ab = np.transpose(img_Lab[:, :, 1:3], (2, 0, 1))

        Y_batch.append(img_Lab[:, :, 0])
        ab_batch.append(np.transpose(img_Lab[:, :, 1:3], (2, 0, 1)))

    temp = np.array(ab_batch)

    ab_batch = temp / 255
    Y_batch = (np.array(Y_batch) - 50) / 100

    Y_batch = torch.from_numpy(Y_batch).double()
    Y_batch = Y_batch.view(-1, 1, 32, 32)

    # ab_batch2 = np.transpose(ab_batch, (2, 0, 1))

    ab_batch = torch.from_numpy(ab_batch).double()
    # ab_batch = ab_batch.view(-1, 2, 32, 32)

    # Standarization:
    # image = (image - mean) / std
    # mean = ab_batch.mean()
    # std = ab_batch.std()
    #
    #
    # means = ab_batch.mean(dim=1, keepdim=True)
    # stds = ab_batch.std(dim=1, keepdim=True)

    return Y_batch, ab_batch


if __name__ == "__main__":
    main()
