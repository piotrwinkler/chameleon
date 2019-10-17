# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Check how YUV were normalized in paper
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_models import FCN_net1
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from skimage import color
from image_colorization.cifar_dataset_class import CifarDataset
from image_colorization.configuration import *


results_dir = f"results/{which_version}"


def main():

    if do_save_results:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    cifar_dataset = CifarDataset(dataset_path, train=choose_train_dataset, ab_preprocessing=ab_chosen_normalization,
                                 L_processing=L_chosen_normalization, kernel_size=gauss_kernel_size,
                                 do_blur=False)

    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    net = chosen_net
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
            if L_chosen_normalization == "normalization":
                L_original = L_original*100 + 50
            elif L_chosen_normalization == "standardization":
                L_original = L_original * cifar_dataset.L_std[0] + cifar_dataset.L_mean[0]

            ab_original = np.transpose(ab_batch[0].numpy(), (1, 2, 0))
            if ab_chosen_normalization == "normalization":
                ab_original = ab_original * 255

            elif ab_chosen_normalization == "standardization":
                ab_original = ab_original * cifar_dataset.ab_std[0] + cifar_dataset.ab_mean[0]

            if plot_lab:
                plt.imshow(np.dstack((L_original, ab_original))[:, :, 0])
                plt.show()

            img_rgb_original = color.lab2rgb(np.dstack((L_original, ab_original)))

            fig = plt.figure(figsize=(14, 7))
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(img_rgb_original)
            ax1.title.set_text('Ground Truth')
            gray = color.rgb2gray(img_rgb_original)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(gray, cmap=plt.get_cmap('gray'))
            ax2.title.set_text('Gray')

            # Gaussian blur
            if do_blur_processing:
                L_blur = np.transpose(L_batch[0].numpy(), (1, 2, 0))
                L_blur = cv2.GaussianBlur(L_blur, gauss_kernel_size, 0)
                # L_blur = np.transpose(L_blur, (2, 0, 1))
                L_blur = torch.from_numpy(L_blur).double()
                ax3 = fig.add_subplot(1, 4, 3)
                ax3.imshow(L_blur)
                ax3.title.set_text('Input L channel')
                L_batch = L_blur.view(-1, 1, 32, 32)

            outputs = net(L_batch)
            loss = criterion(outputs, ab_batch)

            ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
            if ab_chosen_normalization == "normalization":
                ab_outputs = ab_outputs * 255
                scale = max([np.max(ab_outputs), abs(np.min(ab_outputs))])
                ab_outputs = ab_outputs / scale
                ab_outputs = ab_outputs * 80

            elif ab_chosen_normalization == "standardization":
                ab_outputs = ab_outputs * cifar_dataset.ab_std[0] + cifar_dataset.ab_mean[0]

            img_rgb_outputs = color.lab2rgb(np.dstack((L_original, ab_outputs)))
            ax4 = fig.add_subplot(1, 4, 4)
            ax4.imshow(img_rgb_outputs)
            ax4.title.set_text('model output')
            plt.show()

            if do_save_results:
                matplotlib.image.imsave(f"{results_dir}/{i}.png", img_rgb_outputs)


            running_loss = loss.item()

            print(f'[{(i + 1) * batch_size}] loss: {running_loss}')

    print('Finished Testing')


if __name__ == "__main__":
    main()
