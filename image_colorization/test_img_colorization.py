"""Entrypoint for network testing."""
from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import ImageColorizationTester
from image_colorization.data import consts
import os
from loguru import logger as log
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega

results_dir = f"results/{consts.which_version}"


def main():
    log.info("TEST")
    log.info(f"Choosing version {consts.which_version}")

    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)

    if consts.do_trick:
        config_dict['additional_params']['ab_output_processing'] = "trick"

    if config_dict['additional_params']['do_save_results']:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                                    config_dict['additional_params'])

    network = eval(config_dict['net'])()
    tester = ImageColorizationTester(load_net_path=consts.RETRAINING_NET_DIRECTORY, network=network,
                                     dataset=dataset, results_dir=results_dir, config_dict=config_dict)
    tester.test()


if __name__ == "__main__":
    main()
