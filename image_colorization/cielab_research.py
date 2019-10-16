"""
https://pl.wikipedia.org/wiki/CIELab
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color

img_path = 'datasets/red2.png'

"""
Tak długo aż używamy tego samego formatu Lab do kodowania i dekodowania to wszystko powinno być w porządku
"""

def main():
    img = cv2.imread(img_path)
    brightLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # cv2.imshow("Original", img)
    img_matplot = mpimg.imread(img_path)[:, :, :-1]
    brightLAB = cv2.cvtColor(img_matplot, cv2.COLOR_RGB2LAB)

    # plt.imshow(img_matplot)
    # plt.show()

    ch1 = np.full((50, 50), 75.0)
    ch2 = np.full((50, 50), 127.0)
    ch3 = np.full((50, 50), 128.0)

    test_img = np.dstack((ch1, ch2, ch3))
    test_lab = color.lab2rgb(test_img)
    plt.imshow(test_lab)
    plt.show()
    lab = color.rgb2lab(test_lab, illuminant='E')
    print(f"min a: {np.min(lab[:, :, 1])}")
    print(f"max a: {np.max(lab[:, :, 1])}")

    print(f"min b: {np.min(lab[:, :, 2])}")
    print(f"max b: {np.max(lab[:, :, 2])}")

    rgb = io.imread(img_path)
    rgb = rgb / 255.0
    plt.imshow(img_matplot)
    plt.show()
    lab = color.rgb2lab(rgb[:, :, :-1])
    plt.imshow(lab)
    plt.show()
    print(f"min a: {np.min(lab[:, :, 1])}")
    print(f"max a: {np.max(lab[:, :, 1])}")

    print(f"min b: {np.min(lab[:, :, 2])}")
    print(f"max b: {np.max(lab[:, :, 2])}")
    # Lab in skimage:
    """
    L is from 0 to 255
    a and b are from -128 to +127
    """
    rgb2 = color.lab2rgb(lab)
    plt.imshow(rgb2)
    plt.show()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.stack((gray, gray, gray), axis=2)
    # cv2.imshow("gray", gray)

    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Lab in OpenCV:
    """
    L is from 0 to 255
    a and b are from 0 to 255
    """
    cv2.imshow("Lab", img_Lab)
    L = img_Lab[:, :, 0]
    a = img_Lab[:, :, 1]
    b = img_Lab[:, :, 2]

    # a = a - 200
    # b = b + 180
    # L = L - 100

    cv2.imshow("L", L)
    cv2.imshow("a", a)
    cv2.imshow("b", b)

    new_Lab = np.stack((L, a, b), axis=2)
    cv2.imshow("new Lab", new_Lab)

    new_bgr = cv2.cvtColor(new_Lab, cv2.COLOR_LAB2BGR)
    cv2.imshow("new original", new_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
