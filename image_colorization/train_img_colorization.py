"""Entrypoint for network training."""
# TODO: Dopisać tester dla img colorization, dopisać tester dla pojedycznych obrazków
# TODO: Tester dla datasetu
# TODO: Dopisać przekształcenia standardyzacji
import image_colorization.data.consts as consts

from base_classes.setup_creator import SetupCreator
from base_classes.json_parser import JsonParser
from base_classes.trainer import Trainer


def main():
    config_dict = JsonParser.read_config(consts.TRAINING_PARAMETERS)
    dataset = SetupCreator.create_img_color_dataset(consts.TRAINING_DATASET_DIRECTORY, config_dict['dataset'])

    network = consts.chosen_net
    trainer = Trainer(config_dict, consts.NET_SAVING_DIRECTORY, consts.OPTIMIZER_SAVING_DIRECTORY,
                      consts.SCHEDULER_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY, network,
                      consts.RETRAINING_NET_DIRECTORY, consts.RETRAINING_OPTIMIZER_DIRECTORY,
                      consts.RETRAINING_SCHEDULER_DIRECTORY, consts.do_retrain)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
