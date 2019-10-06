from image_colorization.data_server import load_cifar_10
from image_colorization.nets.fcn_model import FCN_net
import torch
import torch.nn as nn
import time
import torch.optim as optim

dataset_path = "image_colorization/datasets/Cifar-10"
save_net_file = "image_colorization/weights/fcn_modelV1.pth"
how_many_epochs = 2
do_load_model = 0
load_net_file = "image_colorization/weights/fcn_modelV1.pth"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if str(device) != "cuda:0":
        raise Exception("No cuda")

    trainloader, testloader, _ = load_cifar_10(dataset_path)

    net = FCN_net()
    if do_load_model == 1:
        net.load_state_dict(torch.load(load_net_file))

    net.train()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        end_time = time.time() - start_time
        print(f"Epoch {epoch} took {end_time} seconds")
        torch.save(net.state_dict(), save_net_file)

    print('Finished Training')

    torch.save(net.state_dict(), save_net_file)


if __name__ == "__main__":
    main()
