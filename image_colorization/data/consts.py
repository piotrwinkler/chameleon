"""This file contains all necessary files directories."""

which_version = "Final"
do_trick = True
choose_test_set = False

load_model = f"{which_version}_epoch_final"

TRAINING_PARAMETERS = f"data/configs/training_parameters_{which_version}.json"
TEST_PARAMETERS = f"data/configs/test_parameters_{which_version}.json"

TRAINING_DATASET_DIRECTORY = '/home/piotr/venvs/inz/project/chameleon/dataset/colour'    # This path has to be overwritten
TEST_DATASET_DIRECTORY = '/home/piotr/venvs/inz/project/chameleon/dataset/colour'   # This path has to be overwritten

NET_SAVING_DIRECTORY = f"model_states/{which_version}/fcn_model{which_version}.pth"
OPTIMIZER_SAVING_DIRECTORY = f"model_states/{which_version}/fcn_optimizer{which_version}.pth"
SCHEDULER_SAVING_DIRECTORY = f"model_states/{which_version}/fcn_scheduler{which_version}.pth"

RETRAINING_NET_DIRECTORY = f"model_states/{which_version}/fcn_model{load_model}.pth"
RETRAINING_OPTIMIZER_DIRECTORY = f"model_states/{which_version}/fcn_optimizer{load_model}.pth"
RETRAINING_SCHEDULER_DIRECTORY = f"model_states/{which_version}/fcn_scheduler{load_model}.pth"

TENSORBOARD_DIRECTORY = "."
