"""
https://pl.wikipedia.org/wiki/CIELab
"""

import cv2
import numpy as np

img_path = 'datasets/lena_color.jpg'


def main():
    img = cv2.imread(img_path)
    cv2.imshow("Original", img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.stack((gray, gray, gray), axis=2)
    # cv2.imshow("gray", gray)

    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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
