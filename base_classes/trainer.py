import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time

from loguru import logger as log
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """This is main training class. It uses parameters directly from "training_parameters.json".
    It allows to monitor training steps, save NN models and use gpu to speed up computations. """
    def __init__(self, config_dict, network, tensorboard_directory='', net_saving_directory='',
                 optimizer_saving_directory='', scheduler_saving_directory='', retraining_net_directory='',
                 retraining_optimizer_directory='', retraining_scheduler_directory=''):
        self._config_dict = config_dict

        self._tensorboard_directory = tensorboard_directory
        self._net_saving_directory = net_saving_directory
        self._optimizer_saving_directory = optimizer_saving_directory
        self._scheduler_saving_directory = scheduler_saving_directory

        self._retraining_net_directory = retraining_net_directory
        self._retraining_optimizer_directory = retraining_optimizer_directory
        self._retraining_scheduler_directory = retraining_scheduler_directory

        self._writer = SummaryWriter()

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else
                                    'cpu')
        log.info(self._device)

        # collecting data from config file
        try:
            self._train_on_gpu = config_dict['train_on_gpu']
            self._network = network.to(self._device) if self._train_on_gpu else network
            self._do_retrain = config_dict['retrain']

            self._criterion = getattr(nn, self._config_dict['criterion']['name'], 'Specified loss criterion not found') \
                (**self._config_dict['criterion']['patameters'])
            log.debug(f'Trainer loss function: {self._criterion}')

            self._optimizer = getattr(optim, self._config_dict['optimizer']['name'], 'Specified optimizer not found') \
                (self._network.parameters(), **self._config_dict['optimizer']['parameters'])
            log.debug(f'Trainer optimizer: {self._optimizer}')

            try:
                self._scheduler = getattr(optim.lr_scheduler, self._config_dict['scheduler']['name'],
                                          'Specified scheduler not found')(optimizer=self._optimizer,
                                                                           **self._config_dict['scheduler'][
                                                                               'parameters'])
                log.debug(f'Trainer scheduler: {self._scheduler}')
            except Exception as e:
                log.debug(f'Cannot initialize scheduler: {e}')
                self._scheduler = None

            self._scheduler_decay_period = self._config_dict['scheduler']['scheduler_decay_period'] \
                if self._scheduler is not None else None
            self._scheduler_decay = self._config_dict['scheduler']['scheduler_decay']  \
                if self._scheduler is not None else None

            self._checker = self._config_dict['training_monitoring_period']
            self._saving_period = self._config_dict['saving_period']

            self._batch_size = self._config_dict['dataloader_parameters']["batch_size"]
            self._init_epoch = self._config_dict["init_epoch"]
            self._training_epochs = self._config_dict['training_epochs']

            self._dataloader_parameters = self._config_dict['dataloader_parameters']
        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)

        # collecting retraining data
        if self._do_retrain:
            try:
                self._network.load_state_dict(torch.load(self._retraining_net_directory))
                log.debug(f'{self._retraining_net_directory} model loaded for retraining')
            except FileNotFoundError:
                log.debug(f'{self._retraining_net_directory}: given retraining model not found! '
                          f'Network will be initialized with random weights.')

            try:
                self._optimizer.load_state_dict(torch.load(self._retraining_optimizer_directory))
                log.debug(f'{self._retraining_optimizer_directory} optimizer loaded for retraining')
            except FileNotFoundError:
                log.debug(f'{self._retraining_net_directory}: given retraining optimizer not found! It will be '
                          f'initialized with random weights.')

            if self._scheduler is not None:
                try:
                    self._scheduler.load_state_dict(torch.load(self._retraining_scheduler_directory))
                    log.info(f'{self._retraining_scheduler_directory} scheduler loaded for retraining')
                except FileNotFoundError:
                    log.debug(f'{self._retraining_net_directory}: given retraining scheduler not found! It will be '
                              f'initialized with random weights.')

        # self._network.train()

    def train(self, dataset):
        dataloader = DataLoader(dataset, **self._dataloader_parameters)

        # main training loop
        for epoch in range(self._init_epoch,
                           self._init_epoch+self._training_epochs):
            running_loss = 0.0
            start_time = time.time()

            for i, data in enumerate(dataloader, 0):
                inputs, expected_outputs = data
                if self._train_on_gpu:
                    inputs = inputs.to(self._device)
                    expected_outputs = expected_outputs.to(self._device)

                self._optimizer.zero_grad()
                outputs = self._network(inputs)

                loss = self._criterion(outputs, expected_outputs)
                loss.backward()
                self._optimizer.step()
                if self._scheduler is not None:
                    self._scheduler.step()

                self._writer.add_scalar(f'{self._tensorboard_directory}/Loss/train', loss.item(), i+1)
                running_loss += loss.item()
                if i % self._checker == self._checker-1:
                    training_loss = running_loss/self._checker
                    log.info(f"[EPOCH: {epoch + 1}, STEP: {i+1}, DATA: {(i+1) * self._batch_size}] "
                             f"loss: {training_loss}")
                    running_loss = 0.0

            end_time = time.time() - start_time
            log.info(f"Epoch {epoch + 1} took {end_time} seconds")

            if self._scheduler is not None and epoch % self._scheduler_decay_period == 0:
                self._scheduler.base_lrs = [self._scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
                self._scheduler.last_epoch = 0
                self._scheduler.step_size = self._scheduler.step_size / self._scheduler_decay
                self._scheduler._step_count = 1

            if epoch % self._saving_period == 0:
                self._save_model(self._optimizer, self._scheduler, title=str(epoch))

            self._save_model(self._optimizer, self._scheduler)
            log.info('Finished Training')
            self._writer.close()

    def _save_model(self, optimizer, scheduler, title="final"):
        try:
            splitted = self._net_saving_directory.split('.')
            torch.save(self._network.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            splitted = self._optimizer_saving_directory.split('.')
            torch.save(optimizer.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            if scheduler is not None:
                splitted = self._scheduler_saving_directory.split('.')
                torch.save(scheduler.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            log.info('NN model weights saved successfully!')
        except Exception as e:
            log.error(f'Trainer was unable to save image due to: {e}')
