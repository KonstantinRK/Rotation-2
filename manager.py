import os
import shutil
from utils import BaseClass
import pandas as pd


class FCMManager(BaseClass):

    _NAME = "fcm"
    _DATASETS = "datasets"
    # _NETWORKS = "networks"
    _EXPERIMENTS = "experiments"

    def __init__(self, experiment=None, root="", *args, **kwargs):
        super(FCMManager, self).__init__(root, *args, **kwargs)
        self.experiment = experiment
        self._setup()

    def setup_experiment(self):
        self.experiment.setup()

    def preprocess_experiment(self):
        self.experiment.preprocess()

    def initialise_experiment(self, experiment, network_dir=None):
        self.experiment = experiment(root=self.get_path(self._EXPERIMENTS))
        self.experiment.initialise(dataset_root=self.get_path(self._DATASETS),
                                   network_root=network_dir)

    def run_experiment(self, overwrite=False):
        self.experiment.run(overwrite)

    def analyse_experiment(self):
        self.experiment.analyse()

    def export_experiment(self, path):
        self.experiment.export(path)

    def get_results(self):
        return self.experiment.get_results()

    def reset_experiment(self, datasets=False):
        self.experiment.reset(datasets)

    def clean_experiment(self):
        self.experiment.clean()

    def export_results(self, path):
        self.experiment.export_results(path)