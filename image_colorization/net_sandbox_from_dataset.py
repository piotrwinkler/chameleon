from torchframe.setup_creator import SetupCreator
from torchframe.json_parser import JsonParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAINING_PARAMETERS = f"data/configs/training_parameters_sandbox.json"


def main():
    config_dict = JsonParser.read_config(TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset('datasets/Cifar-10/cifar-10-batches-py', config_dict['dataset'],
                                                    config_dict['additional_params'])

    fine_elems = [53, 54, 64, 85, 99, 145, 177, 233, 187, 190, 225, 289, 337, 331]

    subplots_x = 7
    subplots_y = 2
    fig = plt.figure(figsize=(16, 8))
    for i, elem in enumerate(fine_elems):
        ax1 = fig.add_subplot(subplots_y, subplots_x, i+1)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        ax1.imshow(dataset.rgb_images[elem])

    plt.show()

    for i, img in enumerate(dataset.rgb_images):
        if i < 221:
            continue
        plt.imshow(img)
        plt.title(str(i))
        plt.show()

    print("End")


if __name__ == "__main__":
    main()
