import sys
sys.path.append('C:/STUDIA/INZYNIERKA/chameleon')
from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import ImageColorizationTester
import os
from loguru import logger as log
from image_colorization.data import consts
import time
import argparse
from image_colorization.nets.fcn_models import *


def main():
    args = parse_args()
    version = args.version
    log.debug(f"Generating results for version: {version}")

    TEST_PARAMETERS = f"data/configs/test_parameters_{version}.json"
    if not os.path.isfile(TEST_PARAMETERS):
        log.error(f"No file {TEST_PARAMETERS}")
        raise Exception(f"No file {TEST_PARAMETERS}")
    RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{version}_epoch_final.pth"
    if not os.path.isfile(RETRAINING_NET_DIRECTORY):
        log.error(f"No file {RETRAINING_NET_DIRECTORY}")
        raise Exception(f"No file {RETRAINING_NET_DIRECTORY}")

    time_start = time.time()
    generate_results(version)
    log.debug(f"It took {time.time() - time_start} seconds")

    log.info("Finished Generating")


def generate_results(version):
    results_dir = f"results/{version}"

    log.info(f"Choosing version {version}")
    TEST_PARAMETERS = f"data/configs/test_parameters_{version}.json"
    RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{version}_epoch_final.pth"

    config_dict = JsonParser.read_config(TEST_PARAMETERS)
    config_dict['additional_params']['do_save_results'] = True
    config_dict['additional_params']['do_show_results'] = False

    if consts.do_trick:
        config_dict['additional_params']['ab_output_processing'] = "trick"

    if config_dict['additional_params']['do_save_results']:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    if consts.choose_test_set:
        config_dict['additional_params']['choose_train_set'] = False

    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                          config_dict['additional_params'])

    network = eval(config_dict['net'])()
    tester = ImageColorizationTester(load_net_path=RETRAINING_NET_DIRECTORY, network=network,
                                     dataset=dataset, results_dir=results_dir, config_dict=config_dict)

    tester.test()


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
