from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net
import torch
from base_classes.logger_class import Logger
import sys
import matplotlib.pyplot as plt
import numpy as np
import torchvision

dataset_path = "image_colorization/datasets/Cifar-10"
load_net_file = "weights/fcn_modelV1.pth"
log_file = "logs/logs_fcn_modelV1_test.log"
how_many_tests = 3


def main():
    # sys.stdout = Logger(log_file)

    _, testloader, _ = load_cifar_10(dataset_path)

    net = FCN_net()
    net.load_state_dict(torch.load(load_net_file))

    net.eval()

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        for _ in range(how_many_tests):
            images, labels = dataiter.next()
            imshow(torchvision.utils.make_grid(images))
            """Convertion to YUV"""
            UV = net(Y)
            """Convertion to RGB"""
            imshow(torchvision.utils.make_grid(RGB_images))


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
