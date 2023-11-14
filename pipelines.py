import cv2
import numpy as np
import time
import shutil
import os
from utils import BaseClass
import tensorflow as tf
from visualiser import plot_activation, plot_all_activations
import pandas as pd
import extractor as ce
import pickle
from monitors import Monitor


class PipeLine(BaseClass):
    _NAME = "pipe"
    _TEMP = "temp"
    _RES = "res"
    _RES_FILE = "res.pkl"

    def __init__(self, network, dataset, root="", *args, **kwargs):
        super(PipeLine, self).__init__(root=root)
        self.network = network
        self.dataset = dataset
        self.activations = None
        self.results = []

    def clean(self):
        if os.path.exists(self.get_path()):
            for i in os.listdir(self.get_path()):
                if i != self._RES:
                    shutil.rmtree(self.get_path(i))

    def _should_overwrite(self, overwrite):
        if os.path.exists(self.get_result_path()) and not overwrite:
            return False
        else:
            return True

    def load(self):
        self.network.load(self.dataset)

    def predict(self, dataset, persist_activations=False):
        if persist_activations:
            self.activations = self.get_activations(dataset)
            predictions = self.activations[-1]
        else:
            test_data = dataset.load_approx_all_test()
            predictions = np.concatenate([self.network.predict(d) for d in test_data])
        return predictions

    def get_activations(self, dataset, layer=None):
        test_data = dataset.load_all_test()
        if layer is None:
            layer_activations = [[] for _ in range(len(self.network.get_layers()))]
            for d in test_data:
                act = self.network.activations(d)
                for i, a in enumerate(act):
                    layer_activations[i].append(a)
            return [np.concatenate(a) for a in layer_activations]
        else:
            return np.concatenate([self.network.activations(d, layer) for d in test_data])

    def get_gradients(self, activation):
        pass

    def learn_concepts(self, concept, control_concept):
        pass

    def run(self, overwrite=True):
        if self._should_overwrite(overwrite):
            res = self._run(overwrite)
            self.store_results()
            return res

    def _run(self, *args, **kwargs):
        pass

    def store_results(self):
        os.makedirs(self.get_path(self._RES), exist_ok=True)
        self._store_results()

    def _store_results(self):
        path = self.get_result_path(self._RES + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def get_result_path(self, file_name=None):
        if file_name is None:
            return self.get_path(self._RES)
        else:
            return os.path.join(self.get_path(self._RES), file_name)

    def get_results(self):
        pass

    def preprocess(self):
        pass


class TrainPipeLine(PipeLine):
    _NAME = "train"

    def __init__(self, network, dataset, root="", epochs=15, *args, **kwargs):
        super(TrainPipeLine, self).__init__(network, dataset, root, *args, **kwargs)
        self.network = network
        self.dataset = dataset
        self.epochs = epochs
        self.activations = None
        self.results = []

    def _get_steps(self, data):
        return len(data) / self.dataset.batch_size

    def _default_train_parameters(self, train_data, val_data):
        return {"x": train_data, "epochs": self.epochs, "steps_per_epoch": self._get_steps(train_data),
                "validation_data": val_data, "validation_steps": self._get_steps(val_data)}

    def train(self, force=False):
        if force or not os.path.exists(self.network.get_path(self.dataset.get_name())):
            d_train = self.dataset.load_train()
            d_val = self.dataset.load_val()
            self.network.fit(**self._default_train_parameters(d_train, d_val))
            self.network.save(self.dataset.get_name())

    def load(self):
        res = self.network.load(self.dataset)
        if not res:
            self.train()

    def _run(self, overwrite=True, *args, **kwargs):
        pass

    def preprocess(self, overwrite=True):
        self.train(overwrite)
        self.load()


class TCAVPipeLine(PipeLine):
    _NAME = "tcav"
    _RES_FILE = "results.csv"

    def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
                 num_exp=500, alpha=0.1, split=False, root="", *args, **kwargs):
        super(TCAVPipeLine, self).__init__(network, dataset, root, *args, **kwargs)
        self._init_tcav(network, dataset, concepts, control_concepts, targets, layers, max_examples,
                        num_exp, alpha, split, self.get_path())

    def _init_tcav(self, network, dataset, concepts, control_concepts, targets, layers, max_examples, num_exp, alpha,
                   split, root):
        self.tcav = ce.TCAVBaseExtractor(network=network, dataset=dataset, concepts=concepts,
                                         control_concepts=control_concepts, targets=targets, layers=layers,
                                         max_examples=max_examples, num_exp=num_exp, alpha=alpha, split=split,
                                         root=root)

    def _run(self, *args, **kwargs):
        res, header = self.tcav.run_tcav()
        self.results = {"header": header, "results": res}
        self.store_results()

    def _store_results(self, header=None):
        df = pd.DataFrame(self.results["results"], columns=self.results["header"])
        df.to_csv(self.get_result_path(self._RES_FILE))

    def get_results(self, as_dataframe=True):
        df = pd.read_csv(self.get_result_path(self._RES_FILE), index_col=0)
        if as_dataframe:
            return df
        else:
            return {"header": list(df.columns), "results": df.to_list()}


class ModTCAVPipeLine(TCAVPipeLine):
    _NAME = "mod_tcav"

    def _init_tcav(self, network, dataset, concepts, control_concepts, targets, layers, max_examples, num_exp, alpha,
                   split, root):
        self.tcav = ce.ModTCAVBaseExtractor(network=network, dataset=dataset, concepts=concepts,
                                            control_concepts=control_concepts, targets=targets, layers=layers,
                                            max_examples=max_examples, num_exp=num_exp, alpha=alpha, split=split,
                                            root=root)


class TestTCAVPipeLine(TCAVPipeLine):
    _NAME = "mod_tcav"

    def _init_tcav(self, network, dataset, concepts, control_concepts, targets, layers, max_examples, num_exp, alpha,
                   split, root):
        self.tcav = ce.TCAVCavExtractor(network=network, dataset=dataset, concepts=concepts,
                                        control_concepts=control_concepts, targets=targets, layers=layers,
                                        max_examples=max_examples, num_exp=num_exp, alpha=alpha, split=split,
                                        root=root)


class MonitorPipeLine(PipeLine):
    _NAME = "monitor_pipe"

    def __init__(self, network, dataset, monitor, monitor_dataset, validation_foo=None, root="", batches=100,
                 batch_size=None, *args, **kwargs):
        super(MonitorPipeLine, self).__init__(network, dataset, root, *args, **kwargs)
        self.monitor = monitor
        self.monitor_dataset = monitor_dataset
        self.batches = batches
        self.batch_size = self.dataset.batch_size if batch_size is None else batch_size
        self.validation_foo = (
            lambda x, y, z: len(set(y).intersection(set(z))) / len(z)) if validation_foo is None else validation_foo

    def load(self):
        super(MonitorPipeLine, self).load()
        self.monitor.load()

    def preprocess(self):
        self.monitor.preprocess()

    def _validate(self, predictions, results, labels):
        print("-" * 100)
        val = []
        for i in range(len(labels)):
            res = results[i]
            pred = predictions[i]
            lab = self.monitor_dataset.translate_label(labels[i])
            # print(res, lab)
            # print(res, "|", lab, self.validation_foo(res, lab))
            val.append(self.validation_foo(pred, res, lab))
        return val

    def validate(self):
        for i in range(len(self.results["results"])):
            self._validate(self.results["results"][i], self.results["labels"][i])

    def _run(self, *args, **kwargs):
        self.results = {"results": [], "labels": [], "validation": [], "accuracy": [], "prediction": []}
        test_data = self.monitor_dataset.load_test(batch_size=self.batch_size, batches=self.batches, with_labels=True)
        batches = min(len(test_data), self.batches)
        n = 0
        for d in test_data:
            if n == self.batches:
                break
            pred, res = self.monitor.monitor(d[0], with_judgements=True, labels=[self.monitor_dataset.translate_label(i) for i in d[1]])
            res = list(res)
            pred = list(pred)
            lab = list(d[1])
            # print(pred)
            # print("X"*100)
            # print(res)
            # print("X" * 100)
            # print(lab)
            # print("X" * 100)
            # print("X" * 100)
            # print("")
            val = self._validate(pred, res, lab)
            acc = np.average(val)
            self.results["results"].append(res)
            self.results["labels"].append(lab)
            self.results["validation"].append(val)
            self.results["accuracy"].append(acc)
            self.results["prediction"].append(pred)
            n += 1
            print("Batch: {0} / {1};  Batch Size: {2};  Accuracy: {3}".format(n, self.batches, self.batch_size, acc))
        # self.validate()
        print("#" * 100)
        print("Accuracy:", np.average(self.results["accuracy"]), "  STD:", np.std(self.results["accuracy"]))
        print("#" * 100)
        return pred


class BaseMonitorPipeLine(MonitorPipeLine):
    _NAME = "base_tcav_monitoring"

    def __init__(self, network, dataset, monitor_dataset, concepts, control_concepts, targets=None, layers=None,
                 max_examples=500, num_exp=500, alpha=0.1, split=False, validation_foo=None, root="",
                 reasoner=None, *args, **kwargs):
        self.root_dir = root
        monitor = self._init_monitor(network=network, dataset=dataset, concepts=concepts,
                                     control_concepts=control_concepts, targets=targets, layers=layers,
                                     max_examples=max_examples, num_exp=num_exp, alpha=alpha, split=split,
                                     root=self.get_path(), reasoner=reasoner)
        super(BaseMonitorPipeLine, self).__init__(network, dataset, monitor, monitor_dataset, validation_foo, root,
                                                      *args, **kwargs)

    @staticmethod
    def _get_extractor_class():
        return ce.TCAVBaseMonitorExtractor

    def _init_monitor(self, network, dataset, concepts, control_concepts, targets, layers, max_examples, num_exp, alpha,
                      split, root, reasoner):
        add_kwargs = self._get_add_kwargs()
        extractor = self._get_extractor_class()(network=network, dataset=dataset, concepts=concepts,
                                             control_concepts=control_concepts, targets=targets, layers=layers,
                                             max_examples=max_examples, num_exp=num_exp, alpha=alpha, split=split,
                                             root=root, reasoner=reasoner, **add_kwargs)
        return Monitor(network, extractor, root)

    @staticmethod
    def _get_add_kwargs():
        return {}


class CavMonitorPipeLine(BaseMonitorPipeLine):
    _NAME = "cav_tcav_monitoring"

    @staticmethod
    def _get_add_kwargs():
        return {}

    @staticmethod
    def _get_extractor_class():
        return ce.TCAVCavExtractor


class ACavMonitorPipeLine(BaseMonitorPipeLine):
    _NAME = "acav_tcav_monitoring"

    @staticmethod
    def _get_add_kwargs():
        return {"use_prediction": False}

    @staticmethod
    def _get_extractor_class():
        return ce.TCAVCavExtractor

#
# class ACavMonitorPipeLine(BaseMonitorPipeLine):
#     _NAME = "acav_tcav_monitoring"
#
#     @staticmethod
#     def _get_add_kwargs():
#         return {"only_relevant": True}
#
#     @staticmethod
#     def _get_extractor_class():
#         return ce.TCAVCavExtractor


class SaliencyMonitorPipeLine(BaseMonitorPipeLine):
    _NAME = "saliency_tcav_monitoring"

    @staticmethod
    def _get_extractor_class():
        return ce.TCAVSaliencyExtractor


class SaliencyMonitorPipeLine2(BaseMonitorPipeLine):
    _NAME = "saliency_tcav_monitoring2"

    @staticmethod
    def _get_extractor_class():
        return ce.TCAVSaliencyExtractor2