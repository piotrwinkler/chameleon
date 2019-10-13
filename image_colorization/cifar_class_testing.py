from image_colorization.cifar_dataset_class import CifarDataset
import torch
import time
cifar_path = 'datasets/Cifar-10/cifar-10-batches-py'


def main():
    start_time = time.time()
    cifar_dataset = CifarDataset(cifar_path, train=True)
    end_time = time.time() - start_time
    print(end_time)
    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=10,
                                              shuffle=False, num_workers=0)

    for i, (x, y) in enumerate(trainloader):
        # print(x)
        # print(y)
        break

    print("end")


if __name__ == "__main__":
    main()
