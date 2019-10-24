from base_classes.json_parser import JsonParser
from base_classes.setup_creator import SetupCreator
from base_classes.tester import ImageColorizationTester
import os
from loguru import logger as log
# from image_colorization.data import consts
import gc

chosen_versions = ["V40", "V40_temp1", "V40_temp2", "V40_temp3"]


def main():

    for version in chosen_versions:
        TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"
        if not os.path.isfile(TRAINING_PARAMETERS):
            raise Exception(f"No file {TRAINING_PARAMETERS}")

    for version in chosen_versions:
        log.info("TRAINING")
        log.info(f"Choosing version {version}")
        NET_SAVING_DIRECTORY = f"model_states/{version}/fcn_model{version}.pth"

        os.makedirs(NET_SAVING_DIRECTORY.rsplit('/', 1)[0], exist_ok=True)
        log.info(f"Saving models to directory {NET_SAVING_DIRECTORY.rsplit('/', 1)[0]}")

        RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{version}_epoch_final.pth"

        config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
        dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'],
                                              config_dict['additional_params'])

        trainer = Trainer(config_dict, consts.NET_SAVING_DIRECTORY, consts.OPTIMIZER_SAVING_DIRECTORY,
                          consts.SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY,
                          consts.RETRAINING_NET_DIRECTORY, consts.RETRAINING_OPTIMIZER_DIRECTORY,
                          consts.RETRAINING_SCHEDULER_DIRECTORY)
        trainer.train(dataset)

        print("Cleaning")
        del dataset, trainer, config_dict
        gc.collect()

    log.info("Finished Generating")


if __name__ == "__main__":
    main()
