# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_models import FCN_net1
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


dataset_path = "datasets/Cifar-10"

which_version = "V8"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"

batch_size = 128


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    if str(device) != "cuda:0":
        raise Exception("No cuda")

    trainloader, testloader, _ = load_cifar_10(path_to_cifar10=dataset_path, batch_size=batch_size)

    net = FCN_net1()
    net = net.double()
    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    net.load_state_dict(torch.load(load_net_file))
    net.eval()
    net.to(device)
    writer = SummaryWriter()

    criterion = nn.MSELoss(reduction='mean').cuda()

    with torch.no_grad():
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data

            Y_batch, ab_batch = yuv_convert(inputs)

            # zero the parameter gradients
            # forward + backward + optimize
            outputs = net(Y_batch.to(device))
            loss = criterion(outputs, ab_batch.to(device))
            writer.add_scalar('Loss/eval', loss)

            running_loss = loss.item()
            # if i % 5 == 4:  # print every 5 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, (i + 1) * batch_size, running_loss / 5))
            #     running_loss = 0.0

            print(f'[{(i + 1) * batch_size}] loss: {running_loss}')
            # break

    print('Finished Evaluating')
    writer.close()


def yuv_convert(imgs_batch):
    Y_batch = []
    ab_batch = []

    for i in range(imgs_batch.shape[0]):
        img_rgb = np.transpose(imgs_batch[i].numpy(), (1, 2, 0))
        # plt.imshow(img_rgb)
        # plt.show()
        img_Lab = color.rgb2lab(img_rgb)

        Y_batch.append(img_Lab[:, :, 0])
        ab_batch.append(img_Lab[:, :, 1:3])

    ab_batch = np.array(ab_batch) / 255
    Y_batch = (np.array(Y_batch) - 50) / 100

    Y_batch = torch.from_numpy(Y_batch).double()
    Y_batch = Y_batch.view(-1, 1, 32, 32)

    ab_batch = torch.from_numpy(ab_batch).double()
    ab_batch = ab_batch.view(-1, 2, 32, 32)

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
