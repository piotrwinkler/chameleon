import numpy as np
import cv2
import glob

img_size = 64


def load_dataset(dataset_dir):
    x, y = [], []

    for img_name in glob.glob(dataset_dir + "*"):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (img_size, img_size))

        Lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = Lab_img[:, :, 0]
        ab = Lab_img[:, :, 1:3]
        # cv2.imshow("original", img)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x.append(L)
        y.append(ab)

    x = np.array(x)
    y = np.array(y)

    x = np.reshape(x, (x.shape[0], img_size, img_size, 1))
    y = np.reshape(y, (y.shape[0], img_size, img_size, 2))

    x = np.array(x)
    y = np.array(y)

    return (x - 256 / 2) / 256, (y - 256 / 2) / 256


def normalize(x_train, x_test, x_validate):
    """
    This module performs normalization of input images

    :param x_train:
        Input training data.

    :param x_test:
        Input testing data.

    :param x_validate:
        Input validation data.

    :return:
        Normalized input images.
    """
    return (x_train - 256 / 2) / 256, (x_test - 256 / 2) / 256, (x_validate - 256 / 2) / 256