"""
https://pl.wikipedia.org/wiki/CIELab
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color

img_path = 'datasets/samolot.jpg'

"""
Tak długo aż używamy tego samego formatu Lab do kodowania i dekodowania to wszystko powinno być w porządku
"""

def main():
    img = cv2.imread(img_path)
    # cv2.imshow("Original", img)
    img_matplot = mpimg.imread(img_path)
    # plt.imshow(img_matplot)
    # plt.show()

    rgb = io.imread(img_path)
    rgb = rgb / 255.0
    plt.imshow(img_matplot)
    plt.show()
    lab = color.rgb2lab(rgb)
    plt.imshow(lab)
    plt.show()
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
