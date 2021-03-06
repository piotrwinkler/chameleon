import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter import filedialog, Tk
from skimage import io, color
from image_colorization.data import consts
from torchframe.json_parser import JsonParser
from image_colorization.nets.fcn_models import *

font_size = 30


def main():
    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)
    if consts.do_trick:
        config_dict['additional_params']['ab_output_processing'] = "trick"
    if consts.choose_test_set:
        config_dict['additional_params']['choose_train_set'] = False

    root = Tk()     # Used to get paths to files from user
    root.withdraw()

    img_paths = filedialog.askopenfilenames(initialdir=".",
                                             title="Select input image",
                                             filetypes=(("All images", "*.*"), ("PNG images", "*.png"),
                                                        ("JPG images", ".jpg"),
                                                        ("JPEG images", "*.jpeg*")))

    net = eval(config_dict['net'])()
    net.load_state_dict(torch.load(consts.RETRAINING_NET_DIRECTORY))

    net.eval()

    cifar_L_mean = np.array([[50.85373370118165]])
    cifar_L_std = np.array([[24.258469539526466]])

    cifar_ab_mean = np.array([[[0.39576409, 5.72532708]]])
    cifar_ab_std = np.array([[[10.15176795, 16.08094785]]])

    for img_path in img_paths:
        rgb_img = io.imread(img_path)
        rgb_img = rgb_img / 255.0
        skladowa_A = color.rgb2lab(rgb_img)[:, :, 1]
        skladowa_B = color.rgb2lab(rgb_img)[:, :, 2]
        gray_img = color.rgb2gray(rgb_img)
        gray_img = np.dstack((gray_img, gray_img, gray_img))
        L_input_gray = color.rgb2lab(gray_img)[:, :, 0]

        L_gray = L_input_gray[:, :, np.newaxis]

        fig2 = plt.figure(figsize=(16, 8))

        if config_dict['additional_params']['L_input_processing'] == "normalization":
            print("Normalization on L_gray channel")
            L_input_gray = (L_input_gray - 50) / 100

        elif config_dict['additional_params']['L_input_processing'] == "standardization":
            print("Standardization on L_gray channel")
            L_input_gray = (L_input_gray - cifar_L_mean) / cifar_L_std

        if config_dict['additional_params']['blur']['do_blur']:
            print(f"Blurring L channel with kernel {config_dict['additional_params']['blur']['kernel_size']}")
            L_input_gray = cv2.GaussianBlur(L_input_gray, tuple(config_dict['additional_params']['blur']['kernel_size']), 0)

        L_batch_gray = torch.from_numpy(L_input_gray).float()
        L_batch_gray = L_batch_gray.view(-1, 1, L_input_gray.shape[0], L_input_gray.shape[1])

        outputs = net(L_batch_gray)
        ab_outputs = np.transpose(outputs[0].detach().numpy(), (1, 2, 0))

        if config_dict['additional_params']['ab_output_processing'] == "normalization":
            ab_outputs = ab_outputs * 255

        elif config_dict['additional_params']['ab_output_processing'] == "standardization":
            ab_outputs = ab_outputs * cifar_ab_std + cifar_ab_mean

        elif config_dict['additional_params']['ab_output_processing'] == "trick":
            scale_L = L_gray / 100
            scale = max([np.max(ab_outputs), abs(np.min(ab_outputs))])
            ab_outputs = ab_outputs / scale
            ab_outputs = ab_outputs * (scale_L * 127)

        img_rgb_outputs = color.lab2rgb(np.dstack((L_gray, ab_outputs)))

        rgb_plot = fig2.add_subplot(1, 1, 1)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        rgb_plot.set_yticklabels([])
        rgb_plot.set_xticklabels([])
        rgb_plot.imshow(img_rgb_outputs)
        rgb_plot.title.set_text(f'')
        change_subplot_fontsize(rgb_plot, font_size)

        plt.show()

    print('Finished Testing')


def change_subplot_fontsize(ax1, desired_size):
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(desired_size)


if __name__ == "__main__":
    main()
