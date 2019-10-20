"""Entrypoint for network training."""
import image_colorization.data.consts as consts

from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from base_classes.trainer import Trainer
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega


def main():
    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'])

    network = FCN_net2()
    trainer = Trainer(config_dict, consts.NET_SAVING_DIRECTORY, consts.OPTIMIZER_SAVING_DIRECTORY,
                      consts.SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY, network,
                      consts.RETRAINING_NET_DIRECTORY, consts.RETRAINING_OPTIMIZER_DIRECTORY,
                      consts.RETRAINING_SCHEDULER_DIRECTORY, consts.do_retrain)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
