"""Entrypoint for network training."""
import data.consts as consts

from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from base_classes.trainer import Trainer
from sobel_filter import SobelFilter


def main():
    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'])

    network = SobelFilter()
    trainer = Trainer(config_dict, consts.NET_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY, network,
                      consts.RETRAINING_NET_DIRECTORY)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
