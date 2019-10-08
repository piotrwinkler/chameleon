# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
# TODO: Proper image format on input
# TODO: Check how images were normalized in paper
from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net
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
dataset_path = "datasets/Cifar-10"

which_version = "V1"

load_net_file = f"model_states/fcn_model{which_version}_epoch{0}.pth"
load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{0}.pth"
load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{0}.pth"

log_file = "logs/logs_fcn_modelV1_train.log"

init_epoch = 0
how_many_epochs = 1
do_load_model = False
batch_size = 100
learning_rate = 1
momentum = 0.9
lr_step_scheduler = 1
lr_step_gamma = 0.9999999
step_decay = 0.5


def main():

    save_net_file = f"model_states/fcn_model{which_version}_epoch{0}.pth"
    save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{0}.pth"
    save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{0}.pth"

    sys.stdout = Logger(log_file)

    trainloader, testloader, _ = load_cifar_10(path_to_cifar10=dataset_path, batch_size=batch_size)

    net = FCN_net()
    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    criterion = nn.MSELoss(reduction='mean')

    # Możliwe, też, że to ma być:
    # criterion = nn.MSELoss(reduction='sum')
    # I wtedy:
    # loss = criterion(outputs, labels) / output.size(0)

    # criterion = nn.SmoothL1Loss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_scheduler, gamma=lr_step_gamma)

    if do_load_model:
        net.load_state_dict(torch.load(load_net_file))
        optimizer.load_state_dict(torch.load(load_optimizer_file))
        scheduler.load_state_dict(torch.load(load_scheduler_file))

    net.train()

    torch.save(net.state_dict(), save_net_file)
    torch.save(optimizer.state_dict(), save_optimizer_file)
    torch.save(scheduler.state_dict(), save_scheduler_file)

    writer = SummaryWriter()

    for epoch in range(init_epoch, init_epoch+how_many_epochs):  # loop over the dataset multiple times

        start_time = time.time()

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data

            Y_batch, ab_batch = yuv_convert(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(Y_batch)
            loss = criterion(outputs, ab_batch)
            writer.add_scalar('Loss/train', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, (i + 1) * batch_size, running_loss / 50))
                running_loss = 0.0

        end_time = time.time() - start_time
        print(f"Epoch {epoch} took {end_time} seconds")

        save_net_file = f"model_states/fcn_modelV1_epoch{0}.pth"
        save_optimizer_file = f"model_states/fcn_optimizerV1_epoch{0}.pth"
        save_scheduler_file = f"model_states/fcn_schedulerV1_epoch{0}.pth"

        torch.save(net.state_dict(), save_net_file)
        torch.save(optimizer.state_dict(), save_optimizer_file)
        torch.save(scheduler.state_dict(), save_scheduler_file)

        if epoch % 25 == 0:
            scheduler.base_lrs = [scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
            scheduler.last_epoch = 0
            scheduler.step_size = scheduler.step_size / step_decay
            scheduler._step_count = 1

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)
    writer.close()


def yuv_convert(imgs_batch):
    Y_batch = []
    ab_batch = []
    # imgs_batch = imgs_batch.view(-1, 32, 32, 3)
    # temp = imgs_batch[0]
    # imgs_batch = np.array(imgs_batch)

    # a = imgs_batch[0]
    # b = np.array(a)
    # a = imgs_batch.shape[0]
    for i in range(imgs_batch.shape[0]):
        img_rgb = np.transpose(imgs_batch[i].numpy(), (1, 2, 0))
        # img_rgb = imgs_batch[i]
        # img_rgb = np.swapaxes(img_rgb, 0, 2)
        # img_rgb = np.flip(img_rgb, axis=0)
        # Obrazy są w RGB
        # r,g, b = img_rgb[0, :, :], img_rgb[1, :, :], img_rgb[2, :, :]
        # img_rgb = np.reshape(img_rgb, (32, 32, 3))
        # img_rgb = np.stack((r, g, b), axis=2)
        # bgr_img = np.stack((b, g, r), axis=2)
        norm_image = cv2.normalize(img_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)

        cv2.imshow("Original", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_Lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        cv2.imshow("Original", img_Lab)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        Y_batch.append(img_Lab[:, :, 0])
        ab_batch.append(img_Lab[:, :, 1:3])

    return Y_batch, ab_batch


if __name__ == "__main__":
    main()
