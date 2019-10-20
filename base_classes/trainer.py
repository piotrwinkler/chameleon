import sys
import torch
import torch.nn as nn
import torch.optim as optim

from loguru import logger as log
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """This is main training class. It uses parameters directly from "training_parameters.json".
    It allows to monitor training steps, save NN models and use gpu to speed up computations. """
    def __init__(self, config_dict, net_saving_directory, tensorboard_directory, network, retraining_network_path=""):
        self._config_dict = config_dict
        self._net_saving_directory = net_saving_directory
        self._tensorboard_directory = tensorboard_directory
        self._retraining_network_path = retraining_network_path
        self._writer = SummaryWriter()

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else
                                    'cpu')
        log.info(self._device)
        try:
            self._train_on_gpu = config_dict['train_on_gpu']
        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)
        self._network = network.to(self._device) if self._train_on_gpu else network

        if self._retraining_network_path != "":
            try:
                self._network.load_state_dict(torch.load(self._retraining_network_path))
                log.info(f'{self._retraining_network_path} model loaded for retraining')
            except FileNotFoundError:
                log.debug(f'{self._retraining_network_path}: given model not found! Network will be initialized '
                          f'with random weights.')

    def train(self, dataset):
        try:
            criterion = getattr(nn, self._config_dict['criterion']['name'], 'Specified loss criterion not found')\
                (**self._config_dict['criterion']['patameters'])
            log.info(f'Trainer loss function: {criterion}')
            optimizer = getattr(optim, self._config_dict['optimizer']['name'], 'Specified optimizer not found')\
                (self._network.parameters(), **self._config_dict['optimizer']['parameters'])
            log.info(f'Trainer optimizer: {optimizer}')

            dataloader = DataLoader(dataset, **self._config_dict['dataloader_parameters'])

            for epoch in range(self._config_dict['training_epochs']):
                running_loss = 0.0
                for i, data in enumerate(dataloader, 0):
                    inputs, expected_outputs = data
                    if self._train_on_gpu:
                        inputs = inputs.to(self._device)
                        expected_outputs = expected_outputs.to(self._device)

                    optimizer.zero_grad()
                    outputs = self._network(inputs)

                    loss = criterion(outputs, expected_outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    checker = self._config_dict['training_monitoring_period']
                    if i % checker == checker-1:
                        training_loss = running_loss/checker
                        log.info(f'EPOCH: {epoch + 1}, STEP: {i+1}] loss: {training_loss}')
                        self._writer.add_scalar(f'{self._tensorboard_directory}/Loss/train', training_loss, i+1)
                        running_loss = 0.0

            self._save_model()
        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)

    def _save_model(self):
        try:
            torch.save(self._network.state_dict(), self._net_saving_directory)
            log.info('NN model weights saved succesfully!')
        except Exception as e:
            log.error(f'Trainer was unable to save image due to: {e}')
