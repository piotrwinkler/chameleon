"""Entrypoint for network training."""
# TODO: Optimizery: Adam, Adagrad, SGE
# TODO: Loss functions: L2, L1 smoothed, L1 (MAE - Mean Absolute Error), Cross Entropy Loss
# TODO: Skrypt do masowego testowania
# TODO: Masowe generowanie do funckji, żeby zwalniać pamięć

import image_colorization.data.consts as consts

from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from base_classes.trainer import Trainer
from loguru import logger as log
import os


def main():
    log.info("TRAINING")
    log.info(f"Choosing version {consts.which_version}")
    os.makedirs(consts.NET_SAVING_DIRECTORY.rsplit('/', 1)[0], exist_ok=True)
    log.info(f"Saving models to directory {consts.NET_SAVING_DIRECTORY.rsplit('/', 1)[0]}")

    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                                    config_dict['additional_params'])

    trainer = Trainer(config_dict, consts.NET_SAVING_DIRECTORY, consts.OPTIMIZER_SAVING_DIRECTORY,
                      consts.SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY,
                      consts.RETRAINING_NET_DIRECTORY, consts.RETRAINING_OPTIMIZER_DIRECTORY,
                      consts.RETRAINING_SCHEDULER_DIRECTORY)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
