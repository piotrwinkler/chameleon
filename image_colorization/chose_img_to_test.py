import torch
import numpy as np
import matplotlib.pyplot as plt
from image_colorization.configuration import *
import cv2
from tkinter import filedialog, Tk
from skimage import io, color


def main():

    root = Tk()     # Used to get paths to files from user
    root.withdraw()

    img_path = filedialog.askopenfilename(initialdir=".",
                                             title="Select input image",
                                             filetypes=(("All images", "*.*"), ("PNG images", "*.png"),
                                                        ("JPG images", ".jpg"),
                                                        ("JPEG images", "*.jpeg*")))


    net = chosen_net
    net = net.double()
    net.load_state_dict(torch.load(load_net_file))
    net.eval()

    # cifar_dataset = CifarDataset(dataset_path, train=choose_train_dataset, ab_preprocessing=ab_chosen_normalization,
    #                              L_processing=L_chosen_normalization, kernel_size=gauss_kernel_size,
    #                              do_blur=do_blur_processing, get_data_to_tests=True)

    cifar_L_mean = np.array([[50.85373370118165]])
    cifar_L_std = np.array([[24.258469539526466]])

    cifar_ab_mean = np.array([[[0.39576409, 5.72532708]]])
    cifar_ab_std = np.array([[[10.15176795, 16.08094785]]])

    with torch.no_grad():
        rgb_img = io.imread(img_path)
        rgb_img = rgb_img / 255.0
        gray_img = color.rgb2gray(rgb_img)
        gray_img = np.dstack((gray_img, gray_img, gray_img))
        L_input_gray = color.rgb2lab(gray_img)[:, :, 0]
        L_gray = L_input_gray[:, :, np.newaxis]

        if L_input_processing == "normalization":
            print("Normalization on L_gray channel")
            L_input_gray = (L_input_gray - 50) / 100

        elif L_input_processing == "standardization":
            print("Standardization on L_gray channel")
            L_input_gray = (L_input_gray - cifar_L_mean) / cifar_L_std

        if L_blur_processing:
            print(f"Blurring L channel with kernel {gauss_kernel_size}")
            L_input_gray = cv2.GaussianBlur(L_input_gray, gauss_kernel_size, 0)

        L_batch_gray = torch.from_numpy(L_input_gray).double()
        L_batch_gray = L_batch_gray.view(-1, 1, L_input_gray.shape[0], L_input_gray.shape[1])

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

        # if L_chosen_normalization == "normalization":
        #     L_gray = L_gray * 100 + 50
        #
        # elif L_chosen_normalization == "standardization":
        #     L_gray = L_gray * cifar_dataset.L_std[0] + cifar_dataset.L_mean[0]

        ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
        if ab_output_processing == "normalization":
            ab_outputs = ab_outputs * 255

        elif ab_output_processing == "standardization":
            ab_outputs = ab_outputs * cifar_ab_std + cifar_ab_mean

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

    print('Finished Testing')


if __name__ == "__main__":
    main()
