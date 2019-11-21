import numpy as np
import pickle
from PIL import Image
import os

dataset_directory = 'datasets/Cifar-10/cifar-10-batches-py'
output_dataset_directory = 'datasets/cifar10_images'


def main():
    print("Loading train set")
    for i in range(1, 6):
        cifar_data_dict = unpickle(dataset_directory + "/data_batch_{}".format(i))
        if i == 1:
            rgb_images = cifar_data_dict[b'data']
        else:
            rgb_images = np.vstack((rgb_images, cifar_data_dict[b'data']))

    print("Loading test set")
    cifar_data_dict = unpickle(dataset_directory + "/test_batch")
    rgb_images = np.vstack((rgb_images, cifar_data_dict[b'data']))

    rgb_images = rgb_images.reshape((len(rgb_images), 3, 32, 32))
    rgb_images = np.rollaxis(rgb_images, 1, 4)

    print("Dataset loaded")
    print("Saving dataset to images")

    for i, image in enumerate(rgb_images):
        im = Image.fromarray(image)
        im.save(os.path.join(output_dataset_directory, f"img_{i:05}.png"))
        if i % 1000 == 0:
            print(f"Saved {i} images")

    print("Finished saving CIFAR-10 to png images")


def unpickle(file):
    with open(file, 'rb') as pickle_file:
        data = pickle.load(pickle_file, encoding='bytes')
    return data


if __name__=="__main__":
    main()
