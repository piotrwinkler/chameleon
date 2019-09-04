import json
from abc import ABC, abstractmethod

import base_classes.consts as consts
import keras


class FilterSketch(ABC):
    def __init__(self):
        self.config = self._read_config()
        self.batch_size = self.config["batch_size"],
        self.epochs = self.config["epochs"],
        self.verbose = self.config["verbose"],
        self.callbacks = self.config["callbacks"],
        self.validation_split = self.config["validation_split"],
        self.validation_data = self.config["validation_data"],
        self.shuffle = self.config["shuffle"],
        self.class_weight = self.config["class_weight"],
        self.sample_weight = self.config["sample_weight"],
        self.initial_epoch = self.config["initial_epoch"],
        self.steps_per_epoch = self.config["steps_per_epoch"],
        self.validation_steps = self.config["validation_steps"],
        self.validation_freq = self.config["validation_freq"]

    @abstractmethod
    def prepare_model(self):
        pass

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def set_validation_split(self, validation_split):
        self.validation_split = validation_split

    def set_validation_data(self, validation_data):
        self.validation_data = validation_data

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def set_class_weight(self, class_weight):
        self.class_weight = class_weight

    def set_sample_weight(self, sample_weight):
        self.sample_weight = sample_weight

    def set_initial_epoch(self, initial_epoch):
        self.initial_epoch = initial_epoch

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def set_validation_steps(self, validation_steps):
        self.validation_steps = validation_steps

    def set_validation_freq(self, validation_freq):
        self.validation_freq = validation_freq

    def _read_config(self) -> dict:
        with open(consts.CONFIG, "r") as config_file:
            config = json.load(config_file)
        return config


class SomeFIlter(FilterSketch):
    def __init__(self):
        super().__init__()

    def prepare_model(self):
        print("something")


f = SomeFIlter()
print(f.config)
print(f.epochs)
f.set_epochs(100)
print(f.epochs)
