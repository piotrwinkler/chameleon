"""This file contains all necessary files directories."""

which_version = "V1"

TRAINING_PARAMETERS = f"data/configs/training_parameters_{which_version}.json"
TEST_PARAMETERS = "data/test_parameters.json"

TRAINING_DATASET_DIRECTORY = 'datasets/Cifar-10/cifar-10-batches-py'    # This path has to be overriten
TEST_DATASET_DIRECTORY = 'datasets/Cifar-10/cifar-10-batches-py'   # This path has to be overriten

NET_SAVING_DIRECTORY = f"model_states/fcn_model{which_version}.pth"
OPTIMIZER_SAVING_DIRECTORY = f"model_states/fcn_optimizer{which_version}.pth"
SCHEDULER_SAVING_DIRECTORY = f"model_states/fcn_scheduler{which_version}.pth"

do_retrain = False
retrain_model = "V1_epoch_final"
RETRAINING_NET_DIRECTORY = f"model_states/fcn_model{retrain_model}.pth"
RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/fcn_optimizer{retrain_model}.pth"
RETRAINING_SCHEDULER_DIRECTORY = f"model_states/fcn_scheduler{retrain_model}.pth"

TENSORBOARD_DIRECTORY = "."

get_data_to_tests = False
choose_train_set = True
