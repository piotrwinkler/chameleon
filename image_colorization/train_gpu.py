# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Check how YUV were normalized in paper
# TODO: Y by Gaussian kernel
# TODO: Dataset class for Cifar 10 test set
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net1
import torch
import torch.nn as nn
import time
import torch.optim as optim
from base_classes.logger_class import Logger
import sys
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from skimage import io, color
from image_colorization.cifar_dataset_class import CifarDataset


dataset_path = 'datasets/Cifar-10/cifar-10-batches-py'

which_version = "V13"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"

log_file = f"logs/logs_fcn_model{which_version}_train.log"

init_epoch = 0
how_many_epochs = 5
do_load_model = False

batch_size = 128
learning_rate = 0.01
momentum = 0.9
lr_step_scheduler = 1
lr_step_gamma = 0.99
step_decay = 0.5
decay_after_steps = 20


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    if str(device) != "cuda:0":
        raise Exception("No cuda")

    save_net_file = f"model_states/fcn_model{which_version}_epoch{0}.pth"
    save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{0}.pth"
    save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{0}.pth"

    # sys.stdout = Logger(log_file)

    cifar_dataset = CifarDataset(dataset_path, train=True, preprocessing="normalization", do_blur=True)
    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    net = FCN_net1()
    net = net.double()
    net.train()

    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    if do_load_model:
        net.load_state_dict(torch.load(load_net_file))

    # net.train()
    net.to(device)

    criterion = nn.MSELoss(reduction='mean').cuda()

    # Możliwe, też, że to ma być:
    # criterion = nn.MSELoss(reduction='sum')
    # I wtedy:
    # loss = criterion(outputs, labels) / output.size(0)

    # criterion = nn.SmoothL1Loss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_scheduler, gamma=lr_step_gamma)

    if do_load_model:
        optimizer.load_state_dict(torch.load(load_optimizer_file))
        scheduler.load_state_dict(torch.load(load_scheduler_file))

    torch.save(net.state_dict(), save_net_file)
    torch.save(optimizer.state_dict(), save_optimizer_file)
    torch.save(scheduler.state_dict(), save_scheduler_file)

    writer = SummaryWriter()

    for epoch in range(init_epoch, init_epoch+how_many_epochs):  # loop over the dataset multiple times

        start_time = time.time()

        for i, data in enumerate(trainloader):
            L_batch, ab_batch = data

            L_batch = L_batch.view(-1, 1, 32, 32)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(L_batch.to(device))
            loss = criterion(outputs, ab_batch.to(device))
            writer.add_scalar('Loss/train', loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print statistics
            running_loss = loss.item()

            print(f'[{epoch + 1}, {(i + 1) * batch_size}] loss: {running_loss}')

        end_time = time.time() - start_time
        print(f"Epoch {epoch + 1} took {end_time} seconds")

        save_net_file = f"model_states/fcn_model{which_version}_epoch{0}.pth"
        save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{0}.pth"
        save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{0}.pth"

        torch.save(net.state_dict(), save_net_file)
        torch.save(optimizer.state_dict(), save_optimizer_file)
        torch.save(scheduler.state_dict(), save_scheduler_file)

        if epoch % decay_after_steps == 0:
            scheduler.base_lrs = [scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
            scheduler.last_epoch = 0
            scheduler.step_size = scheduler.step_size / step_decay
            scheduler._step_count = 1

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)
    writer.close()


if __name__ == "__main__":
    main()
