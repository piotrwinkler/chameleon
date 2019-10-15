from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import Tester
from data import consts
from sobel_filter import SobelFilter


def main():
    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)

    Tester.test_img_img_network(**SetupCreator.create_testbase(consts.DATASET_DIRECTORY, consts.NET_SAVING_DIRECTORY,
                                                               SobelFilter, config_dict))


if __name__ == "__main__":
    main()
