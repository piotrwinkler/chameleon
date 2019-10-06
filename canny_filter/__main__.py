import data.consts as consts
import torch

from base_classes.general_dataset import GeneralDataset
from base_classes.trainer import Trainer
from base_classes.dummy_converter_to_furure_removal import ImagesConverter
from base_classes.transforms import ToTensor, Rescale, Normalize
from canny_filter import CannyFIlter
from torchvision import transforms

conversion_parameters = ['mean', 100, 200]
dataset = GeneralDataset(consts.DATASET_DIRECTORY, ImagesConverter.canny_filter, conversion_parameters,
                         transform=transforms.Compose([Rescale((32, 32)),
                                                       Normalize(),
                                                       ToTensor()]))
network = CannyFIlter().to(dtype=torch.float64)
trainer = Trainer(consts.TRAINING_PARAMETERS, consts.SAVING_DIRECTORY, network)
trainer.train(dataset)
