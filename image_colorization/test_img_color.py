"""Entrypoint for network testing."""
from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import ImageColorizationTester
from image_colorization.data import consts
import os

results_dir = f"results/{consts.which_version}"


def main():
    if consts.do_save_results:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'])

    tester = ImageColorizationTester(consts.RETRAINING_NET_DIRECTORY, consts.chosen_net, dataset,
                                     config_dict['dataloader_parameters'], config_dict['do_save_results'])
    tester.test()


if __name__ == "__main__":
    main()
