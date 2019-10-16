from image_colorization.cifar_dataset_class import CifarDataset
import torch
import time
from image_colorization.configuration import *


def main():
    start_time = time.time()
    cifar_dataset = CifarDataset(dataset_path, train=choose_train_dataset, ab_preprocessing=ab_chosen_normalization,
                                 L_processing=L_chosen_normalization, kernel_size=gauss_kernel_size,
                                 do_blur=do_blur_processing, get_data_to_tests=False)
    end_time = time.time() - start_time
    print(end_time)
    trainloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=10,
                                              shuffle=False, num_workers=0)

    for i, (L, ab) in enumerate(trainloader):
        # print(x)
        # print(y)
        break

    print("end")


if __name__ == "__main__":
    main()
