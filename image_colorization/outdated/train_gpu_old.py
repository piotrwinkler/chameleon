# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Check how YUV were normalized in paper
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


dataset_path = "datasets/Cifar-10"

which_version = "V7"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"

log_file = f"logs/logs_fcn_model{which_version}_train.log"

init_epoch = 0
how_many_epochs = 20
do_load_model = False

batch_size = 32
learning_rate = 0.1
momentum = 0.9
lr_step_scheduler = 1
lr_step_gamma = 0.9
step_decay = 0.5


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

    trainloader, testloader, _ = load_cifar_10(path_to_cifar10=dataset_path, batch_size=batch_size)

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

        # running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data

            Y_batch, ab_batch = yuv_convert(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(Y_batch.to(device))
            loss = criterion(outputs, ab_batch.to(device))
            writer.add_scalar('Loss/train', loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print statistics
            running_loss = loss.item()
            # if i % 5 == 4:  # print every 5 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, (i + 1) * batch_size, running_loss / 5))
            #     running_loss = 0.0

            print(f'[{epoch + 1}, {(i + 1) * batch_size}] loss: {running_loss}')
            # break

        end_time = time.time() - start_time
        print(f"Epoch {epoch} took {end_time} seconds")

        save_net_file = f"model_states/fcn_model{which_version}_epoch{0}.pth"
        save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{0}.pth"
        save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{0}.pth"

        torch.save(net.state_dict(), save_net_file)
        torch.save(optimizer.state_dict(), save_optimizer_file)
        torch.save(scheduler.state_dict(), save_scheduler_file)

        if epoch % 100 == 0:
            scheduler.base_lrs = [scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
            scheduler.last_epoch = 0
            scheduler.step_size = scheduler.step_size / step_decay
            scheduler._step_count = 1

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)
    writer.close()

    # exit()


def yuv_convert(imgs_batch):
    Y_batch = []
    ab_batch = []

    for i in range(imgs_batch.shape[0]):
        img_rgb = np.transpose(imgs_batch[i].numpy(), (1, 2, 0))
        # plt.imshow(img_rgb)
        # plt.show()
        img_Lab = color.rgb2lab(img_rgb)

        Y_batch.append(img_Lab[:, :, 0])
        ab_batch.append(np.transpose(img_Lab[:, :, 1:3], (2, 0, 1)))

    ab_batch = np.array(ab_batch) / 255
    Y_batch = (np.array(Y_batch) - 50) / 100

    Y_batch = torch.from_numpy(Y_batch).double()
    Y_batch = Y_batch.view(-1, 1, 32, 32)

    ab_batch = torch.from_numpy(ab_batch).double()
    # ab_batch = ab_batch.view(-1, 2, 32, 32)

    # Standarization:
    # image = (image - mean) / std
    # mean = ab_batch.mean()
    # std = ab_batch.std()
    #
    #
    # means = ab_batch.mean(dim=1, keepdim=True)
    # stds = ab_batch.std(dim=1, keepdim=True)

    return Y_batch, ab_batch


if __name__ == "__main__":
    main()
