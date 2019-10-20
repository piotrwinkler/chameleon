# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from skimage import color
from image_colorization.cifar_dataset_class import CifarDataset
from image_colorization.configuration import *
import cv2

results_dir = f"results/{which_version}"
# results_dir = f"results/V72_2"


def main():

    if do_save_results:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    cifar_dataset = CifarDataset(dataset_path, train_set=choose_train_dataset, ab_preprocessing=ab_input_processing,
                                 L_processing=L_input_processing, kernel_size=gauss_kernel_size,
                                 do_blur=L_blur_processing, get_data_to_tests=True)

    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    net = chosen_net
    net = net.double()
    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    net.load_state_dict(torch.load(load_net_file))
    net.eval()
    print(f"Choosing net fcn_model{which_version}_epoch{which_epoch_version}")

    criterion = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for i, (_, ab_batch_rgb, rgb_images, L_batch_gray, gray_images) in enumerate(trainloader):
            rgb_img = rgb_images[0]
            gray_img = gray_images[0]
            L_gray = np.transpose(L_batch_gray[0].numpy(), (1, 2, 0))
            L_input_gray = L_gray[:, :, 0]

            if L_blur_processing:
                print(f"Blurring L channel with kernel {gauss_kernel_size}")
                L_input_gray = cv2.GaussianBlur(L_input_gray, gauss_kernel_size, 0)
                L_batch_gray = torch.from_numpy(L_input_gray).double()
                L_batch_gray = L_batch_gray.view(-1, 1, 32, 32)

            fig = plt.figure(figsize=(14, 7))
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(rgb_img)
            ax1.title.set_text('Ground Truth')

            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(gray_img)
            ax2.title.set_text('Gray')

            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(L_input_gray)
            ax3.title.set_text(f'gray L channel, blur={L_blur_processing}')

            outputs = net(L_batch_gray)
            loss = criterion(outputs, ab_batch_rgb)

            if L_input_processing == "normalization":
                L_gray = L_gray * 100 + 50

            elif L_input_processing == "standardization":
                L_gray = L_gray * cifar_dataset.L_std[0] + cifar_dataset.L_mean[0]

            ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
            if ab_output_processing == "normalization":
                ab_outputs = ab_outputs * 255

            elif ab_output_processing == "standardization":
                ab_outputs = ab_outputs * cifar_dataset.ab_std[0] + cifar_dataset.ab_mean[0]

            elif ab_output_processing == "trick":
                scale_L = L_gray / 100
                scale = max([np.max(ab_outputs), abs(np.min(ab_outputs))])
                ab_outputs = ab_outputs / scale
                ab_outputs = ab_outputs * (scale_L * 127)

            img_rgb_outputs = color.lab2rgb(np.dstack((L_gray, ab_outputs)))

            ax4 = fig.add_subplot(1, 4, 4)
            ax4.imshow(img_rgb_outputs)
            ax4.title.set_text('model output')
            if do_show_results:
                plt.show()

            if do_save_results:
                matplotlib.image.imsave(f"{results_dir}/{str(i).zfill(4)}.png", img_rgb_outputs)

            running_loss = loss.item()

            print(f'[{(i + 1) * batch_size}] loss: {running_loss}')

            if i == how_many_results_to_generate:
                break

    print('Finished Testing')


if __name__ == "__main__":
    main()
