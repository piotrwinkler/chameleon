"""Entrypoint for network training"""
import data.consts as consts

from base_classes.dataset_classes import BasicFiltersDataset
from base_classes.trainer import Trainer
from base_classes.dummy_converter_to_furure_removal import ImagesConverter
from base_classes.transforms import ToTensor, Rescale, Normalize
from canny_filter import CannyFIlter
from torchvision import transforms

# PREPROCESSING CONFIGURATION
# conversion_parameters = {'threshold_type': 'mean', 'threshold_lower': 100, 'threshold_upper': 200}  # Parameters used
conversion_parameters = {}
# during input image filtering by given ImagesConverter
dataset = BasicFiltersDataset(consts.DATASET_DIRECTORY, ImagesConverter.sobel_filter, conversion_parameters,
                         transform=transforms.Compose([Rescale((200, 200)),     # define input image size here
                                                       Normalize(),
                                                       ToTensor()]))
network = CannyFIlter()
trainer = Trainer(consts.TRAINING_PARAMETERS, consts.NET_SAVING_DIRECTORY, consts.TENSORBOARD_DIRECTORY, network)
trainer.train(dataset)
