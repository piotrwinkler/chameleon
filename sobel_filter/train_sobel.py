"""Entrypoint for network training."""
import data.consts as consts

from torchframe.setup_creator import SetupCreator
from torchframe.json_parser import JsonParser
from torchframe.trainer import Trainer
from sobel_filter import SobelFilter


def main():
    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                          config_dict['additional_params'])

    network = eval(config_dict['net_model'])()
    trainer = Trainer(config_dict, network, tensorboard_directory=consts.TENSORBOARD_DIRECTORY,
                      net_saving_directory=consts.NET_SAVING_DIRECTORY,
                      retraining_net_directory=consts.RETRAINING_NET_DIRECTORY,
                      optimizer_saving_directory=consts.OPTIMIZER_SAVING_DIRECTORY)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
