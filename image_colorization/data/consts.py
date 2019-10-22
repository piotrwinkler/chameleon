"""This file contains all necessary files directories."""
from image_colorization.nets.fcn_models import FCN_net1, FCN_net2, FCN_net3, FCN_net4, FCN_net5, FCN_net_mega

which_version = "V70_2"
chosen_net = FCN_net_mega()
do_retrain = True
load_model = f"{which_version}_epoch_final"

TRAINING_PARAMETERS = f"data/configs/training_parameters_{which_version}.json"
TEST_PARAMETERS = f"data/configs/test_parameters_{which_version}.json"

TRAINING_DATASET_DIRECTORY = 'datasets/Cifar-10/cifar-10-batches-py'    # This path has to be overriten
TEST_DATASET_DIRECTORY = 'datasets/Cifar-10/cifar-10-batches-py'   # This path has to be overriten

NET_SAVING_DIRECTORY = f"model_states/fcn_model{which_version}.pth"
OPTIMIZER_SAVING_DIRECTORY = f"model_states/fcn_optimizer{which_version}.pth"
SCHEDULER_SAVING_DIRECTORY = f"model_states/fcn_scheduler{which_version}.pth"

RETRAINING_NET_DIRECTORY = f"model_states/fcn_model{load_model}.pth"
RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/fcn_optimizer{load_model}.pth"
RETRAINING_SCHEDULER_DIRECTORY = f"model_states/fcn_scheduler{load_model}.pth"

TENSORBOARD_DIRECTORY = "."
