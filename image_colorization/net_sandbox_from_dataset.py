from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TRAINING_PARAMETERS = f"data/configs/training_parameters_sandbox.json"


def main():
    config_dict = JsonParser.read_config(TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset('datasets/Cifar-10/cifar-10-batches-py', config_dict['dataset'],
                                                    config_dict['additional_params'])

    # dataloader = DataLoader(dataset, **config_dict['dataloader_parameters'])
    fine_elems = [53, 54, 64, 85, 99, 145, 177, 233, 187, 190, 225, 289, 337, 331]


    subplots_x = 7
    subplots_y = 2
    fig = plt.figure(figsize=(16, 8))
    for i, elem in enumerate(fine_elems):
        ax1 = fig.add_subplot(subplots_y, subplots_x, i+1)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        # ax1.xticks(ticks=[])
        # ax1.yticks(ticks=[])        # ax1.Tick.remove
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        ax1.imshow(dataset.rgb_images[elem])

    # plt.xticks(ticks=[])
    # plt.yticks(ticks=[])
    plt.show()


    # for elem in fine_elems:
    #     plt.imshow(dataset.rgb_images[elem])
    #     plt.title(str(elem))
    #     plt.show()

    for i, img in enumerate(dataset.rgb_images):
        if i<221:
            continue
        # print(x)
        # print(y)
        plt.imshow(img)
        plt.title(str(i))
        plt.show()

    # 185+ chyba żaba spoko
    # 190 spoko statek, 221 chyba też



    print("End")


if __name__ == "__main__":
    main()
