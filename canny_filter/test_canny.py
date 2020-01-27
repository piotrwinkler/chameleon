"""Entrypoint for network testing."""
from torchframe.json_parser import JsonParser
from torchframe.setup_creator import SetupCreator
from torchframe.tester import TestImgtoImg
from data import consts
from canny_filter import CannyFilter


def main():
    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)

    network = eval(config_dict['net_model'])
    tester = TestImgtoImg(**SetupCreator.create_testbase(consts.TEST_DATASET_DIRECTORY, consts.NET_LOADING_DIRECTORY,
                                                         network, config_dict))
    tester.test()


if __name__ == "__main__":
    main()
