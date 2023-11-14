from pprint import pprint
import tensorflow as tf
from utils import BaseClass
import datetime
import tf_explain as tfe
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback


class Monitor(BaseClass):

    _NAME = "monitor"

    def __init__(self, network, extractor, root=""):
        super(Monitor, self).__init__(root)
        self.network = network
        self.extractor = extractor
        self.judgements = None

    def preprocess(self):
        self.extractor.preprocess()
        self._preprocess()

    def _preprocess(self):
        pass

    def judge(self, extract):
        self.judgements = extract

    def monitor(self, data, with_judgements=True, labels=None):
        self.extractor.tcavs_as_table()
        predictions, extract = self.extractor.extract(data, with_extract=True, labels=labels)
        self.judge(extract)
        if with_judgements:
            return predictions, self.judgements
        else:
            return predictions

    def get_judgements(self):
        return self.judgements

    def load(self):
        self.extractor.load()


class ReasonMonitor(Monitor):

    def __init__(self, network, extractor, contradictory_concepts,  root=""):
        super(Monitor, self).__init__(network=network, extractor=extractor, root=root)