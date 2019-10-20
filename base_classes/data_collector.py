import glob
import sys

from loguru import logger as log


class DataCollector:
    """This class gathers methods intended to collect data from specified directories."""
    def __init__(self):
        pass

    @staticmethod
    def collect_images(dataset_directory) -> list:
        """
        :param dataset_directory: path to images dataset
        :return: list of images paths
        """
        file_types = [f'{dataset_directory}/*.jpg', f'{dataset_directory}/*.jpeg',
                      f'{dataset_directory}/*.png', f'{dataset_directory}/*.bmp']
        list_of_files_lists = [glob.glob(e) for e in file_types if glob.glob(e) != []]
        files_list = []
        for l in list_of_files_lists:
            files_list = [img for img in l]

        log.info(f'List of loaded files: {files_list}')
        log.info(f'{len(files_list)} images loaded from: {dataset_directory}!')
        if not files_list:
            log.error(f'Images loading from {dataset_directory} failed!')
            sys.exit(1)

        return files_list
