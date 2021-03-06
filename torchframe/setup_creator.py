import torchframe.dataset_classes as datasets
import torchframe.conversions as conversions
import torchframe.transforms as predefined_transforms
import sys

from loguru import logger as log
from torchvision import transforms


class SetupCreator:
    def __init__(self):
        pass

    @staticmethod
    def create_dataset(dataset_directory, dataset_config_dict, additional_params) -> object:
        """Contols flow of the data from "dataset" part of training_parameters.json and fits it to BaseDataset class.
        Should be used in training entrypoint."""
        try:
            dataset = getattr(datasets, dataset_config_dict['name'], 'Specified dataset not found')

            input_conversions_list = [getattr(conversions, conversion['name'],
                                              'Specified input conversion not found')(*conversion['parameters'])
                                      for conversion in dataset_config_dict['input_conversions']]

            output_conversions_list = [getattr(conversions, conversion['name'],
                                       'Specified output conversion not found')(*conversion['parameters'])
                                       for conversion in dataset_config_dict['output_conversions']]

            transforms_list = [getattr(predefined_transforms, transform['name'],
                                       'Specified transform not found')(*transform['parameters'])
                               for transform in dataset_config_dict['transforms']]
        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)
        except TypeError as e:
            log.error(f'Requested parameter is out of scope: {e}')
            sys.exit(1)

        log.info('All dataset features collected properly!')
        return dataset(dataset_directory, input_conversions_list, output_conversions_list, additional_params,
                       transform=transforms.Compose(transforms_list))

    @staticmethod
    def create_testbase(dataset_directory, load_net_path, model, test_config_dict) -> dict:
        """Contols flow of the data from test_parameters.json and fits it to BaseTester class.
        Should be used in testing entrypoint."""
        setup_dict = dict()
        try:
            setup_dict['dataset_directory'] = dataset_directory
            setup_dict['load_net_path'] = load_net_path
            setup_dict['model'] = model()

            setup_dict['transforms'] = [getattr(predefined_transforms, transform['name'],
                                                'Specified transform not found')(*transform['parameters'])
                                        for transform in test_config_dict['transforms']]

            setup_dict['input_conversions_list'] = [getattr(conversions, conversion['name'],
                                                    'Specified input conversion not found')(*conversion['parameters'])
                                                    for conversion in test_config_dict['input_conversions']]

            setup_dict['output_conversions_list'] = [getattr(conversions, conversion['name'],
                                                     'Specified output conversion not found')(*conversion['parameters'])
                                                     for conversion in test_config_dict['output_conversions']]

            setup_dict['additional_params'] = test_config_dict['additional_params']
        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)
        except TypeError as e:
            log.error(f'Requested parameter is out of scope: {e}')
            sys.exit(1)

        log.info('All testbase features collected properly!')
        return setup_dict
