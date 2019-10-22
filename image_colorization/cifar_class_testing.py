import time
import image_colorization.data.consts as consts
from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from torch.utils.data import DataLoader


def main():
    start_time = time.time()

    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                                    config_dict['additional_params'])
    
    end_time = time.time() - start_time
    print(end_time)
    dataloader = DataLoader(dataset, **config_dict['dataloader_parameters'])

    for i, (L, ab) in enumerate(dataloader):
        # print(x)
        # print(y)
        break

    print("end")


if __name__ == "__main__":
    main()
