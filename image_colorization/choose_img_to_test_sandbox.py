import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter import filedialog, Tk
from skimage import io, color
from image_colorization.data import consts
from base_classes.json_parser import JsonParser
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

    # fig = plt.figure(figsize=(14, 7))


    # fig3 = plt.figure(figsize=(14, 7))
    # ax_for_fig3 = fig3.add_subplot(1, 1, 1)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    #
    # ax1 = fig.add_subplot(1, 3, 1)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # ax2 = fig.add_subplot(1, 3, 2)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # # ax3 = fig2.add_subplot(1, 1, 1)
    # ax4 = fig.add_subplot(1, 3, 3)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
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
        # rgb_plot = fig2.add_subplot(1, 1, 1)
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # rgb_plot.set_yticklabels([])
        # rgb_plot.set_xticklabels([])
        # rgb_plot.imshow(rgb_img)
        # rgb_plot.title.set_text(f'')
        # change_subplot_fontsize(rgb_plot, font_size)

        # L_plot = fig2.add_subplot(1, 1, 1)
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # L_plot.set_yticklabels([])
        # L_plot.set_xticklabels([])
        # pos = L_plot.imshow(L_input_gray, cmap="coolwarm")
        # L_plot.title.set_text(f"Składowa L obrazu w formacie CIELab")
        # change_subplot_fontsize(L_plot, font_size)
        # fig2.colorbar(pos, ax=L_plot)




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


        # ax1.imshow(rgb_img)
        # ax1.title.set_text('Obraz rzeczywisty')
        # ax1.set_yticklabels([])
        # ax1.set_xticklabels([])
        # change_subplot_fontsize(ax1, font_size)
        # # ax2 = fig.add_subplot(1, 4, 2)
        # ax2.imshow(gray_img)
        # ax2.title.set_text('Obraz czarno-biały')
        # ax2.set_yticklabels([])
        # ax2.set_xticklabels([])
        # change_subplot_fontsize(ax2, font_size)


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

        # ax4 = fig.add_subplot(1, 4, 4)
        # ax4.imshow(img_rgb_outputs)
        # ax4.title.set_text('Otrzymany rezultat')
        # ax4.set_yticklabels([])
        # ax4.set_xticklabels([])
        # change_subplot_fontsize(ax4, font_size)

        rgb_plot = fig2.add_subplot(1, 1, 1)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        rgb_plot.set_yticklabels([])
        rgb_plot.set_xticklabels([])
        rgb_plot.imshow(img_rgb_outputs)
        rgb_plot.title.set_text(f'')
        change_subplot_fontsize(rgb_plot, font_size)

        #
        # ax_for_fig3.imshow(img_rgb_outputs)
        # ax_for_fig3.title.set_text('Bez zastosowanie przetwarzania końcowego')
        # ax_for_fig3.set_yticklabels([])
        # ax_for_fig3.set_xticklabels([])
        # change_subplot_fontsize(ax_for_fig3, font_size)
        # del net
        # del img_rgb_outputs, ab_outputs, outputs, L_batch_gray, L_gray, L_input_gray, gray_img, rgb_img
        # del config_dict, fig, ax1, ax2, ax3, ax4
        # if additional_params['do_show_results']:
        plt.show()
        # plt.close()
        # del fig

    print('Finished Testing')


def change_subplot_fontsize(ax1, desired_size):
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(desired_size)


if __name__ == "__main__":
    main()
