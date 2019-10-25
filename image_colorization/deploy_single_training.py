import sys
sys.path.append('C:/STUDIA/INZYNIERKA/chameleon')
from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.trainer import Trainer
import os
from loguru import logger as log
from image_colorization.data import consts
import gc
import argparse


def main():
    args = parse_args()
    version = args.version
    log.debug(f"Training version: {version}")

    TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"
    if not os.path.isfile(TRAINING_PARAMETERS):
        log.error(f"No file {TRAINING_PARAMETERS}")
        raise Exception(f"No file {TRAINING_PARAMETERS}")

    deploy_training(version)
    # log.info("Cleaning")
    # gc.collect()

    log.info("Finished Generating")


def deploy_training(version):
    log.info("TRAINING")
    log.info(f"Choosing version {version}")
    NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"

    os.makedirs(NET_SAVING_DIRECTORY.rsplit('/', 1)[0], exist_ok=True)
    log.info(f"Saving models to directory {NET_SAVING_DIRECTORY.rsplit('/', 1)[0]}")

    TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"

    config_dict = JsonParser.read_config(TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                          config_dict['additional_params'])

    load_model = f"{version}_epoch_final"
    NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"
    OPTIMIZER_SAVING_DIRECTORY = f"model_states/{version}/fcn_optimizer{version}.pth"
    SCHEDULER_SAVING_DIRECTORY = f"model_states/{version}/fcn_scheduler{version}.pth"

    RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{load_model}.pth"
    RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/{version}/fcn_optimizer{load_model}.pth"
    RETRAINING_SCHEDULER_DIRECTORY = f"model_states/{version}/fcn_scheduler{load_model}.pth"

    trainer = Trainer(config_dict, NET_SAVING_DIRECTORY, OPTIMIZER_SAVING_DIRECTORY,
                      SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY,
                      RETRAINING_NET_DIRECTORY, RETRAINING_OPTIMIZER_DIRECTORY,
                      RETRAINING_SCHEDULER_DIRECTORY)
    trainer.train(dataset)


def parse_args():
    """
    This module parses command line arguments.
    :return:
        Parser containing command line data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', '-v', help='chosen config version',
                        required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
