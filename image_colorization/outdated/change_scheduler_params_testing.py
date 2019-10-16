from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net1
import torch
import torch.nn as nn
import time
import torch.optim as optim
from base_classes.logger_class import Logger
import sys

dataset_path = "datasets/Cifar-10"
save_net_file = "model_states/fcn_modelV1.pth"
how_many_epochs = 1
do_load_model = 0
load_net_file = "model_states/fcn_modelV1.pth"
log_file = "logs/logs_fcn_modelV1_train.log"
batch_size = 128
learning_rate = 100
momentum = 0.9
lr_step_scheduler = 1
lr_step_gamma = 0.9
step_decay = 0.5


def main():

    # sys.stdout = Logger(log_file)

    trainloader, testloader, _ = load_cifar_10(path_to_cifar10=dataset_path, batch_size=batch_size)

    net = FCN_net1()
    if do_load_model == 1:
        net.load_state_dict(torch.load(load_net_file))

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_scheduler, gamma=lr_step_gamma)

    print(scheduler.state_dict())
    a = optimizer.state_dict()
    # scheduler.state_dict()["step_size"] = 2
    print(scheduler.state_dict())
    print(scheduler.optimizer.state_dict())
    print(scheduler.optimizer.state_dict()["param_groups"][0]["lr"])
    a = scheduler.optimizer.state_dict()["param_groups"]
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    # opt = scheduler.optimizer.state_dict()
    # scheduler.gamma = 0.1
    # scheduler.base_lrs = scheduler.optimizer.param_groups[0]["lr"]
    scheduler.base_lrs = [scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
    scheduler.last_epoch = 0
    scheduler.step_size = 2

    scheduler._step_count = 1
    # scheduler.ste
    # scheduler.optimizer.load_state_dict(opt)
    # optimizer.load_state_dict(opt)
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()

    print(scheduler.state_dict())
    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        start_time = time.time()

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
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
        torch.save(net.state_dict(), save_net_file)

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)


if __name__ == "__main__":
    main()
