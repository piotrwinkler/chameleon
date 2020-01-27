# TODO: Nie działa - używać deplou_multiple_trainings.bat
from torchframe.json_parser import JsonParser
from torchframe.setup_creator import SetupCreator
from torchframe.trainer import Trainer
import os
from loguru import logger as log
from image_colorization.data import consts
import gc
import torch
chosen_versions = ["V40", "V40_temp1", "V40_temp2", "V40_temp3"]


def main():

    raise Exception("Skrypt nie działa")

    for version in chosen_versions:
        TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"
        if not os.path.isfile(TRAINING_PARAMETERS):
            raise Exception(f"No file {TRAINING_PARAMETERS}")

    for version in chosen_versions:
        # torch.cuda.empty_cache()
        deploy_training(version)
        log.info("Cleaning")
        # torch.cuda.empty_cache()

    log.info("Finished Generating")


def deploy_training(version):
    log.info("TRAINING")
    log.info(f"Choosing version {version}")
    NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"

    os.makedirs(NET_SAVING_DIRECTORY.rsplit('/', 1)[0], exist_ok=True)
    log.info(f"Saving models to directory {NET_SAVING_DIRECTORY.rsplit('/', 1)[0]}")

    TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"

    config_dict = JsonParser.read_config(TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                          config_dict['additional_params'])

    load_model = f"{version}_epoch_final"
    NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"
    OPTIMIZER_SAVING_DIRECTORY = f"model_states/{version}/fcn_optimizer{version}.pth"
    SCHEDULER_SAVING_DIRECTORY = f"model_states/{version}/fcn_scheduler{version}.pth"

    RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{load_model}.pth"
    RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/{version}/fcn_optimizer{load_model}.pth"
    RETRAINING_SCHEDULER_DIRECTORY = f"model_states/{version}/fcn_scheduler{load_model}.pth"

    trainer = Trainer(config_dict, NET_SAVING_DIRECTORY, OPTIMIZER_SAVING_DIRECTORY,
                      SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY,
                      RETRAINING_NET_DIRECTORY, RETRAINING_OPTIMIZER_DIRECTORY,
                      RETRAINING_SCHEDULER_DIRECTORY)
    trainer.train(dataset)

    del config_dict, trainer, dataset


if __name__ == "__main__":
    main()
