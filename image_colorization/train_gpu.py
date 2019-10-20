# TODO: Proper Weight initialization
"""
with torch.no_grad():
    self.conv1.weight = torch.nn.Parameter(K)
"""
import torch
import torch.nn as nn
import time
import torch.optim as optim
from base_classes.logger_class import Logger
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from image_colorization.cifar_dataset_class import CifarDataset

from image_colorization.configuration import *


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

    cifar_dataset = CifarDataset(dataset_path, train_set=choose_train_dataset, ab_preprocessing=ab_input_processing,
                                 L_processing=L_input_processing, kernel_size=gauss_kernel_size,
                                 do_blur=L_blur_processing, get_data_to_tests=False)

    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    net = chosen_net
    net = net.double()
    net.train()
    print(f"Choosing net fcn_model{which_version}_epoch{which_epoch_version}")

    # Miało być "per-pixel Euclidean loss function", mam nadzieję, ze to ten MSELoss
    if do_load_model:
        print("Loading wages")
        net.load_state_dict(torch.load(load_net_file))

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

    for epoch in range(init_epoch+1, init_epoch+how_many_epochs+1):

        start_time = time.time()

        for i, (L_batch, ab_batch) in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(L_batch.to(device))
            loss = criterion(outputs, ab_batch.to(device))
            writer.add_scalar('Loss/train', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss = loss.item()

            print(f'[{epoch}, {(i + 1) * batch_size}] loss: {running_loss}')

        end_time = time.time() - start_time
        print(f"Epoch {epoch} took {end_time} seconds")

        if save_every_10_epoch and epoch % 10 == 0:
            print(epoch % 10)
            save_net_file = f"model_states/fcn_model{which_version}_epoch{epoch}.pth"
            save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{epoch}.pth"
            save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{epoch}.pth"
        else:
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

        print(f"Current learning rate = {optimizer.state_dict()['param_groups'][0]['lr']}")

    else:
        save_net_file = f"model_states/fcn_model{which_version}_epoch{epoch}.pth"
        save_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{epoch}.pth"
        save_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{epoch}.pth"

        torch.save(net.state_dict(), save_net_file)
        torch.save(optimizer.state_dict(), save_optimizer_file)
        torch.save(scheduler.state_dict(), save_scheduler_file)

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)
    writer.close()


if __name__ == "__main__":
    main()
