"""Entrypoint for network testing."""
from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import TestImgtoImg
from data import consts
from sepia_filter import SepiaFilter


def main():
    config_dict = JsonParser.read_config(consts.TEST_PARAMETERS)

    tester = TestImgtoImg(**SetupCreator.create_testbase(consts.TEST_DATASET_DIRECTORY, consts.NET_SAVING_DIRECTORY,
                                                         SepiaFilter, config_dict))
    tester.test()


if __name__ == "__main__":
    main()
