from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import ImageColorizationTester
import os
from loguru import logger as log
from image_colorization.data import consts

chosen_versions = ["V70", "V70_2", "V71", "V72", "V73", "V74"]


def main():

    for version in chosen_versions:
        TEST_PARAMETERS = f"data/configs/test_parameters_{version}.json"
        if not os.path.isfile(TEST_PARAMETERS):
            raise Exception(f"No file {TEST_PARAMETERS}")
        RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{version}_epoch_final.pth"
        if not os.path.isfile(RETRAINING_NET_DIRECTORY):
            raise Exception(f"No file {RETRAINING_NET_DIRECTORY}")

    for version in chosen_versions:
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

        tester = ImageColorizationTester(RETRAINING_NET_DIRECTORY, dataset, results_dir, config_dict)

        tester.test()
        del dataset, tester


if __name__ == "__main__":
    main()
