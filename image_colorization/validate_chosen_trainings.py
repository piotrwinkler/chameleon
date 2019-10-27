import sys
sys.path.append('C:/STUDIA/INZYNIERKA/chameleon')
import os
import argparse
from loguru import logger as log
from base_classes.json_parser import JsonParser
from image_colorization.data import consts
from base_classes.trainer import Trainer
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega, \
    FCN_net_mega_V2, FCN_net_mega_dropout


def main():
    args = parse_args()
    version = args.version
    log.debug(f"Validating version: {version}")

    TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"
    if not os.path.isfile(TRAINING_PARAMETERS):
        log.error(f"No file {TRAINING_PARAMETERS}")
        raise Exception(f"No file {TRAINING_PARAMETERS}")
    else:
        config_dict = JsonParser.read_config(TRAINING_PARAMETERS)

        load_model = f"{version}_epoch_final"
        NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"
        OPTIMIZER_SAVING_DIRECTORY = f"model_states/{version}/fcn_optimizer{version}.pth"
        SCHEDULER_SAVING_DIRECTORY = f"model_states/{version}/fcn_scheduler{version}.pth"

        RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{load_model}.pth"
        RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/{version}/fcn_optimizer{load_model}.pth"
        RETRAINING_SCHEDULER_DIRECTORY = f"model_states/{version}/fcn_scheduler{load_model}.pth"

        network = eval(config_dict['net'])()
        trainer = Trainer(config_dict, network=network, tensorboard_directory=consts.TENSORBOARD_DIRECTORY,
                          net_saving_directory=NET_SAVING_DIRECTORY,
                          optimizer_saving_directory=OPTIMIZER_SAVING_DIRECTORY,
                          scheduler_saving_directory=SCHEDULER_SAVING_DIRECTORY,
                          retraining_net_directory=RETRAINING_NET_DIRECTORY,
                          retraining_optimizer_directory=RETRAINING_OPTIMIZER_DIRECTORY,
                          retraining_scheduler_directory=RETRAINING_SCHEDULER_DIRECTORY)
        log.debug(f"Version {version} - OK")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', help='chosen config version',
                        required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
