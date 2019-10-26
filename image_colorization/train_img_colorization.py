"""Entrypoint for network training."""
# TODO: Optimizery: Adam, Adagrad, SGE
# TODO: Loss functions: L2 (inaczej MSELoss), L1 smoothed, L1 (MAE - Mean Absolute Error),
#  Cross Entropy Loss(Nie działa, bo jest dla klasyfikatorów)
import image_colorization.data.consts as consts

from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from base_classes.trainer import Trainer
from loguru import logger as log
import os
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega


def main():
    log.info("TRAINING")
    log.info(f"Choosing version {consts.which_version}")
    os.makedirs(consts.NET_SAVING_DIRECTORY.rsplit('/', 1)[0], exist_ok=True)
    log.info(f"Saving models to directory {consts.NET_SAVING_DIRECTORY.rsplit('/', 1)[0]}")

    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                                    config_dict['additional_params'])

    network = eval(config_dict['net'])()

    trainer = Trainer(config_dict, network=network, tensorboard_directory=consts.TENSORBOARD_DIRECTORY,
                      net_saving_directory=consts.NET_SAVING_DIRECTORY,
                      optimizer_saving_directory=consts.OPTIMIZER_SAVING_DIRECTORY,
                      scheduler_saving_directory=consts.SCHEDULER_SAVING_DIRECTORY,
                      retraining_net_directory=consts.RETRAINING_NET_DIRECTORY,
                      retraining_optimizer_directory=consts.RETRAINING_OPTIMIZER_DIRECTORY,
                      retraining_scheduler_directory=consts.RETRAINING_SCHEDULER_DIRECTORY)

    trainer.train(dataset)


if __name__ == "__main__":
    main()
