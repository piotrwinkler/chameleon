# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Check how YUV were normalized in paper
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net
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

which_version = "V8"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"

batch_size = 1


def main():

    cifar_dataset = CifarDataset(dataset_path)
    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    net = FCN_net()
    net = net.double()
    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    net.load_state_dict(torch.load(load_net_file))
    net.eval()

    criterion = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for i, data in enumerate(trainloader):
            L_batch, ab_batch = data

            L_batch = L_batch.view(-1, 1, 32, 32)

            # img_rgb = np.transpose(L_batch[0].numpy(), (1, 2, 0))
            # plt.imshow(img_rgb)
            # plt.show()

            # lab = color.rgb2lab(img_rgb)

            # L_batch, ab_batch = yuv_convert(inputs)

            L_original = np.transpose(L_batch[0].numpy(), (1, 2, 0))
            L_original = L_original*100 + 50
            ab_original = np.transpose(ab_batch[0].numpy(), (1, 2, 0))
            ab_original = ab_original * 255

            img_rgb_original = color.lab2rgb(np.dstack((L_original, ab_original)))
            # plt.imshow(img_rgb_original)
            # plt.show()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(img_rgb_original)
            gray = color.rgb2gray(img_rgb_original)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(gray, cmap=plt.get_cmap('gray'))

            outputs = net(L_batch)
            loss = criterion(outputs, ab_batch)

            ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
            ab_outputs = ab_outputs * 255

            img_rgb_outputs = color.lab2rgb(np.dstack((L_original, ab_outputs)))
            # plt.imshow(img_rgb_outputs)
            # plt.show()
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(img_rgb_outputs)

            plt.show()

            running_loss = loss.item()

            print(f'[{(i + 1) * batch_size}] loss: {running_loss}')

    print('Finished Testing')


if __name__ == "__main__":
    main()
