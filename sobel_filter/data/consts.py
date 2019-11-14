"""This file contains all necessary files directories."""
TRAINING_PARAMETERS = "data/training_parameters.json"
TEST_PARAMETERS = "data/test_parameters.json"

TRAINING_DATASET_DIRECTORY = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset"    # This path has to be overriten
TEST_DATASET_DIRECTORY = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset"   # This path has to be overriten

NET_SAVING_DIRECTORY = "data/checkpoints/net.pth"
NET_LOADING_DIRECTORY = "data/checkpoints/net_epoch_final.pth"
OPTIMIZER_SAVING_DIRECTORY = "data/checkpoints/optimizer.pth"
SCHEDULER_SAVING_DIRECTORY = ""

RETRAINING_NET_DIRECTORY = "data/checkpoints/new_best.pth"
RETRAINING_OPTIMIZER_DIRECTORY = ""
RETRAINING_SCHEDULER_DIRECTORY = ""

TENSORBOARD_DIRECTORY = "."
