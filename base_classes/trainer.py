import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time

from loguru import logger as log
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega


class Trainer:
    """This is main training class. It uses parameters directly from "training_parameters.json".
    It allows to monitor training steps, save NN models and use gpu to speed up computations. """
    def __init__(self, config_dict, net_saving_directory, optimizer_saving_directory, scheduler_saving_directory,
                 tensorboard_directory, retraining_network_path="", retraining_optimizer_path='',
                 retraining_scheduler_path=''):
        self._config_dict = config_dict
        self._net_saving_directory = net_saving_directory
        self.optimizer_saving_directory = optimizer_saving_directory
        self.scheduler_saving_directory = scheduler_saving_directory
        self._tensorboard_directory = tensorboard_directory
        self._retraining_network_path = retraining_network_path
        self.retraining_optimizer_path = retraining_optimizer_path
        self.retraining_scheduler_path = retraining_scheduler_path
        self.do_retrain = config_dict['retrain']
        self._writer = SummaryWriter()

        network = eval(config_dict['net'])()
        log.debug(f"Choosing net {config_dict['net']}")

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

        if self.do_retrain and self._retraining_network_path != "":
            try:
                self._network.load_state_dict(torch.load(self._retraining_network_path))
                log.debug(f'{self._retraining_network_path} model loaded for retraining')
            except FileNotFoundError:
                log.debug(f'{self._retraining_network_path}: given model not found! Network will be initialized '
                          f'with random weights.')

        # self._network = self._network.double()
        self._network.train()
    def train(self, dataset):
        try:
            criterion = getattr(nn, self._config_dict['criterion']['name'], 'Specified loss criterion not found')\
                (**self._config_dict['criterion']['patameters'])
            log.debug(f'Trainer loss function: {criterion}')

            optimizer = getattr(optim, self._config_dict['optimizer']['name'], 'Specified optimizer not found')\
                (self._network.parameters(), **self._config_dict['optimizer']['parameters'])
            log.debug(f'Trainer optimizer: {optimizer}')

            try:
                scheduler = getattr(optim.lr_scheduler, self._config_dict['scheduler']['name'], 'Specified scheduler not found')\
                    (optimizer=optimizer,
                     **self._config_dict['scheduler']['parameters'])
                log.debug(f'Trainer scheduler: {scheduler}')
            except Exception as e:
                log.debug(f'Cannot initialize scheduler: {e}')
                scheduler = None

            if self.do_retrain and self._retraining_network_path != "":
                try:
                    optimizer.load_state_dict(torch.load(self.retraining_optimizer_path))
                    log.debug(f'{self.retraining_optimizer_path} optimizer loaded for retraining')
                except FileNotFoundError:
                    log.debug(f'{self._retraining_network_path}: given optimizer not found! It will be '
                              f'initialized with random weights.')
                if scheduler is not None:
                    try:
                        scheduler.load_state_dict(torch.load(self.retraining_scheduler_path))
                        log.info(f'{self.retraining_scheduler_path} scheduler loaded for retraining')
                    except FileNotFoundError:
                        log.debug(f'{self._retraining_network_path}: given scheduler not found! It will be '
                                  f'initialized with random weights.')

            dataloader = DataLoader(dataset, **self._config_dict['dataloader_parameters'])
            checker = self._config_dict['training_monitoring_period']

            if scheduler is not None:
                scheduler_decay_period = self._config_dict['scheduler']['scheduler_decay_period']
            else:
                scheduler_decay_period = None

            saving_period = self._config_dict['saving_period']
            batch_size = self._config_dict['dataloader_parameters']["batch_size"]

            for epoch in range(self._config_dict["init_epoch"],
                               self._config_dict["init_epoch"]+self._config_dict['training_epochs']):
                running_loss = 0.0
                start_time = time.time()

                for i, data in enumerate(dataloader, 0):
                    inputs, expected_outputs = data
                    if self._train_on_gpu:
                        inputs = inputs.to(self._device)
                        expected_outputs = expected_outputs.to(self._device)

                    optimizer.zero_grad()
                    outputs = self._network(inputs)

                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    self._writer.add_scalar(f'{self._tensorboard_directory}/Loss/train', loss.item(), i+1)
                    running_loss += loss.item()
                    if i % checker == checker-1:
                        training_loss = running_loss/checker
                        log.info(f"[EPOCH: {epoch + 1}, STEP: {i+1}, DATA: {(i+1) * batch_size}] loss: {training_loss}")
                        log.info(f'EPOCH: {epoch + 1}, STEP: {i+1}] loss: {training_loss}')
                        self._writer.add_scalar(f'{self._tensorboard_directory}/Loss/train', training_loss, i+1)
                        running_loss = 0.0

                end_time = time.time() - start_time
                log.info(f"Epoch {epoch + 1} took {end_time} seconds")

                if scheduler is not None and epoch % scheduler_decay_period == 0:
                    scheduler.base_lrs = [scheduler.optimizer.state_dict()["param_groups"][0]["lr"]]
                    scheduler.last_epoch = 0
                    scheduler.step_size = scheduler.step_size / self._config_dict['scheduler']['scheduler_decay']
                    scheduler._step_count = 1

                if epoch % saving_period == 0:
                    self._save_model(optimizer, scheduler, title=str(epoch))

            self._save_model(optimizer, scheduler)
            log.info('Finished Training')
            self._writer.close()

        except KeyError as e:
            log.error(f'Requested key not found in config dictionary: {e}')
            sys.exit(1)

    def _save_model(self, optimizer, scheduler, title="final"):
        try:
            splitted = self._net_saving_directory.split('.')
            torch.save(self._network.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            splitted = self.optimizer_saving_directory.split('.')
            torch.save(optimizer.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            if scheduler is not None:
                splitted = self.scheduler_saving_directory.split('.')
                torch.save(scheduler.state_dict(), f"{splitted[0]}_epoch_{title}.{splitted[-1]}")

            log.info('NN model weights saved successfully!')
        except Exception as e:
            log.error(f'Trainer was unable to save image due to: {e}')
