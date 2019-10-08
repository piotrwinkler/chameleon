import numpy as np
import torch
import cv2

from base_classes.tester import Tester
from canny_filter import CannyFIlter
from loguru import logger as log


def main():
    img = Tester.read_image('./datasets/test_dataset/kon.jpg')
    img = cv2.resize(img, (200, 200))
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (img - 256 / 2) / 256 if np.mean(img) > 1 \
        else img
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    log.info(img)

    test_conf = {'input_img': img, 'net_path': './data/net.pth', 'model': CannyFIlter()}
    Tester.test_network(**test_conf)


if __name__ == "__main__":
    main()
