from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net
import torch
from base_classes.logger_class import Logger
import sys

dataset_path = "image_colorization/datasets/Cifar-10"
load_net_file = "model_states/fcn_modelV1.pth"
log_file = "logs/logs_fcn_modelV1_eval.log"


def main():
    # sys.stdout = Logger(log_file)

    _, testloader, _ = load_cifar_10(dataset_path)

    net = FCN_net()
    net.load_state_dict(torch.load(load_net_file))

    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            """Calculate loss"""


if __name__ == "__main__":
    main()
