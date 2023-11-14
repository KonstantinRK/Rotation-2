from utils import BaseClass, find_cutoff
import os
from tcav2.model import GenericWrapper
from tcav2.activation_generator import ImageActivationGenerator
from tcav2.utils_plot import plot_results
from tcav2.tcav import TCAV
from tcav2.cav import CAV
from tcav2mod.model import GenericWrapper as ModGenericWrapper
from tcav2mod.activation_generator import ImageActivationGenerator as ModImageActivationGenerator
from tcav2mod.utils_plot import plot_results as mod_plot_results
from tcav2mod.tcav import TCAV as ModTCAV
import pickle
import tensorflow as tf
import shutil
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import sys
import pandas as pd

class Extractor(BaseClass):
    _NAME = "extractor"
    _CONF = "conf"

    def __init__(self, network, root="", clean_dir=False):
        super(Extractor, self).__init__(root)
        self.network = network
        self.config = {}
        self.clean_dir = clean_dir
        self.extracted_values = None

    def get_conf_path(self, file_name=None):
        inp = [self._CONF, file_name] if file_name is not None else self._CONF
        return self.get_path(inp)

    def _write_config(self):
        os.makedirs(self.get_conf_path(), exist_ok=True)
        with tf.io.gfile.GFile(self.get_conf_path(self._CONF + ".pkl"), 'w') as pkl_file:
            pickle.dump(self.config, pkl_file)

    def _load_config(self, config=None):
        if config is None:
            if os.path.exists(self.get_conf_path(self._CONF + ".pkl")):
                # print(self.get_conf_path(self._CONF + ".pkl"))
                with tf.io.gfile.GFile(self.get_conf_path(self._CONF + ".pkl"), 'rb') as pkl_file:
                    self.config = pickle.load(pkl_file)
            else:
                self._make_config_from_temp()
        else:
            self.config = config

    def clean(self):
        if os.path.exists(self.get_path()):
            for i in os.listdir(self.get_path()):
                if i != self._CONF:
                    shutil.rmtree(self.get_path(i))

    def _make_config_from_temp(self):
        pass

    def _compose_config(self):
        pass

    def _preprocess(self):
        pass

    def preprocess(self):
        self._preprocess()
        self._compose_config()
        self._write_config()
        if self.clean_dir:
            self.clean()

    def extract(self, data, with_extract=True, labels=None):
        predictions, self.extracted_values = self._extract(data, labels=labels)
        self._postprocess_extract()
        if with_extract:
            return predictions, self.extracted_values
        else:
            return predictions

    def _postprocess_extract(self):
        pass

    def _extract(self, data, labels=None):
        pass

    def load(self, config=None):
        self._load_config(config)

    def get_extract(self):
        return self.extracted_values


class ActivationExtractor(Extractor):

    def _extract(self, data):
        return self.extract_activations(data)

    def extract_activations(self, data):
        self.network.activations(data)


class ConceptExtractor(ActivationExtractor):
    _NAME = "concept_extractor"

    def __init__(self, network, concepts, root="", clean_dir=False):
        super(ConceptExtractor, self).__init__(network, root, clean_dir)
        self.concepts = concepts


class TCAVBaseExtractor(ConceptExtractor):
    _NAME = "tcav"
    _TEMP = "temp"
    _ACT = os.path.join(_TEMP, 'activations')
    _CAV = os.path.join(_TEMP, 'cavs')
    _RES = "temp_res"

    def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
                 num_exp=500, alpha=0.1, split=False, root="", *args, **kwargs):
        super(TCAVBaseExtractor, self).__init__(network, concepts, root)
        self.dataset = dataset
        self.num_exp = num_exp
        self.targets = targets
        self.max_examples = max_examples
        self.layers = layers
        self.control_concepts = control_concepts
        self.alpha = alpha
        self.results = None
        self.concept_names = sorted(list(j.get_name() for j in concepts))
        self.concept_map = {c: i for i, c in enumerate(self.concept_names)}
        self.rev_concept_map = {i: c for i, c in enumerate(self.concept_names)}
        self.split = split

    def _preprocess(self):
        self.results = self.run_tcav(attach_header=True)

    def _compose_config(self):
        self.config = {"tcavs": self.results}

    def _print_setting(self):
        rand_concepts = [c.get_name() for c in self.control_concepts]
        if len(rand_concepts) == 1 or self.split:
            rand_concepts = [r + "_" + str(i) for i in range(self.num_exp) for r in rand_concepts]
        print("-" * 100)
        print("Settings:")
        print("-" * 100)
        print("->  ", "Network:")
        print("    ", "    ", self.network.get_name())
        print("->  ", "Concepts:")
        for i in [c.get_name() for c in self.concepts]:
            print("    ", "    ", i)
        print("->  ", "Rand. Concepts:")
        for i in rand_concepts:
            print("   ", "    ", i)
        print("-" * 100)

    def buffer_results(self, target, layer, results):
        with open(self.get_path([self._RES, str(target) + "-" + str(layer) + ".pkl"]), "wb") as f:
            pickle.dump(results, f)

    def compute_cav(self, bottleneck, target, clean_output=True):
        tf.compat.v1.logging.set_verbosity(0)
        if len(self.control_concept_names) == 1:
            mytcav = TCAV(target, self.concept_names, [bottleneck], self.act_generator, [self.alpha],
                          cav_dir=self.get_path(self._CAV), num_random_exp=self.num_exp,
                          random_counterpart=self.control_concept_names)  # 10)
        else:
            mytcav = TCAV(target, self.concept_names, [bottleneck], self.act_generator, [self.alpha],
                          cav_dir=self.get_path(self._CAV), num_random_exp=self.num_exp,
                          random_concepts=self.control_concept_names)  # 10)
        results = mytcav.run(run_parallel=False)
        if clean_output:
            output = plot_results(results, num_random_exp=self.num_exp, random_concepts=self.control_concept_names)
            return output[1:], output[0]
        else:
            return results, []

    @staticmethod
    def _import_concept(path, concept, split=None):
        if not os.path.exists(os.path.join(path, concept.get_name())):
            concept.transfer(path, with_labels=False, split=split)

    def setup_tcav(self):
        os.makedirs(self.get_path(self._TEMP), exist_ok=True)
        os.makedirs(self.get_path(self._ACT), exist_ok=True)
        os.makedirs(self.get_path(self._CAV), exist_ok=True)
        os.makedirs(self.get_path(self._RES), exist_ok=True)
        for c in self.concepts:
            self._import_concept(self.get_path(), c)
        for c in self.control_concepts:
            if len(self.control_concepts) == 1 or self.split:
                self._import_concept(self.get_path(), c, split=self.num_exp)
            else:
                self._import_concept(self.get_path(), c)
        for t in self.targets:
            if not os.path.exists(os.path.join(self.get_path(), t)):
                self.dataset.transfer_label(self.get_path(), t)

    def _init_settings(self):
        self.layers = list(range(len(self.network.model.layers) - 1)) if self.layers is None else self.layers
        self.targets = self.dataset.get_labels() if self.targets is None else self.targets
        self.targets = [self.targets] if not isinstance(self.targets, list) else self.targets
        self.concept_names = [c.get_name() for c in self.concepts]
        self.control_concept_names = [c.get_name() for c in self.control_concepts]
        if len(self.control_concept_names) == 1 or self.split:
            self.control_concept_names = [r + "_" + str(i) for i in range(self.num_exp) for r in
                                          self.control_concept_names]
        self.mymodel = GenericWrapper(self.network.model, self.dataset.get_labels(),
                                      shape=self.network.input_size, model_name=self.network.get_name())
        self.act_generator = ImageActivationGenerator(self.mymodel, self.get_path(), self.get_path(self._ACT),
                                                      max_examples=self.max_examples)

    def run_tcav(self, attach_header=False, clean_output=True):
        self._init_settings()
        self.setup_tcav()
        results, header = self._run_tcav(clean_output=clean_output)
        if clean_output:
            if attach_header:
                return [header] + results
            else:
                return results, header
        else:
            return results

    def _run_tcav(self, clean_output=True):
        header = None
        results = [] if clean_output else {}
        self._print_setting()
        for t in self.targets:
            print("#" * 100)
            print("Class", t)
            print("#" * 100)
            for l in self.layers:
                print("-" * 100)
                print("Layer", l)
                print("-" * 100)
                data, header = self.compute_cav(l, t, clean_output)
                self._post_process(t, l)
                if clean_output:
                    results += data
                else:
                    results[(t, l)] = data
                self.buffer_results(t, l, data)
        return results, header

    def clean_raw_results(self, data, header=True):
        results = []
        header_row = []
        for i, k, v in enumerate(data.items()):
            out = plot_results(v, num_random_exp=self.num_exp, random_concepts=self.control_concept_names)
            results += out[1:]
            if i == 0:
                header_row = out[0]
        if header:
            return [header_row] + results
        else:
            return results

    def _load_config(self, config=None):
        self._init_settings()
        super(TCAVBaseExtractor, self)._load_config(config)

    def _make_config_from_temp(self):
        results = {}
        for t in self.targets:
            for l in self.layers:
                with open(self.get_path([self._RES, str(t) + "-" + str(l) + ".pkl"]), "rb") as f:
                    data = pickle.load(f)
                results[(t, l)] = data
                # print(data[0])
        if self.results is None:
            self.results = results
        self.config = results
        self._compose_config()
        # for k,v in self.config.items():
        #     print(k, type(v))
        self._write_config()

    def _post_process(self, target, layer, *args, **kwargs):
        pass


class ModTCAVBaseExtractor(TCAVBaseExtractor):

    def compute_cav(self, bottleneck, target, clean_output=True):
        tf.compat.v1.logging.set_verbosity(0)
        mytcav = ModTCAV(target, self.concept_names, [bottleneck], self.act_generator, [self.alpha],
                         cav_dir=self.get_path(self._CAV), num_random_exp=self.num_exp,
                         random_concepts=self.control_concept_names)  # 10)
        results = mytcav.run(run_parallel=False)
        if clean_output:
            output = mod_plot_results(results, num_random_exp=self.num_exp, random_concepts=self.control_concept_names)
            return output[1:], output[0]
        else:
            return results, []


class TCAVBaseMonitorExtractor(TCAVBaseExtractor):
    _NAME = "base_monitor_extractor"

    _CONF_CAV = "temp_cavs"
    _CONF_ACT = "temp_acts"

    def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
                 num_exp=500, alpha=0.1, split=False, use_prediction=True, only_relevant=False, root="", reasoner=None, *args, **kwargs):
        super(TCAVBaseMonitorExtractor, self).__init__(network=network, dataset=dataset, concepts=concepts,
                                                       control_concepts=control_concepts, targets=targets,
                                                       layers=layers,
                                                       max_examples=max_examples, num_exp=num_exp, alpha=alpha,
                                                       split=split,
                                                       root=root, *args, **kwargs)
        self.use_prediction = use_prediction
        self.reasoner = reasoner
        self.relevant_elements = None
        self.only_relevant = only_relevant

    def _post_process(self, target, layer, *args, **kwargs):
        self._extract_cavs(target, layer)
        self._extract_acts(target, layer)
        try:
            shutil.rmtree(self.get_path(self._CAV))
            shutil.rmtree(self.get_path(self._ACT))
        except FileNotFoundError:
            pass

    def _extract_cavs(self, target, layer):
        base_path = self.get_path([self._CONF_CAV])
        os.makedirs(base_path, exist_ok=True)
        n = len(os.listdir(base_path))
        for i, path in enumerate(os.listdir(self.get_path(self._CAV))):
            with open(self.get_path([self._CAV, path]), "rb") as f:
                cav = pickle.load(f)
            # if cav["concepts"][0] in self.concept_names or cav["concepts"][1] in self.concept_names:
            cav["target"] = target
            cav["layer"] = layer
            cav["concept"] = cav["concepts"][0]
            with open(os.path.join(base_path, str(n + i) + ".pkl"), "wb") as f:
                pickle.dump(cav, f)

    def _extract_acts(self, target, layer):
        base_path = self.get_path([self._CONF_ACT])
        os.makedirs(base_path, exist_ok=True)
        n = len(os.listdir(base_path))
        for i, c in enumerate(self.concept_names + self.control_concept_names + [target]):
            path = "acts_{0}_{1}".format(c, layer)
            acts = {"activations": np.load(self.get_path([self._ACT, path])),
                    "target": target,
                    "layer": layer,
                    "concept": c}
            with open(os.path.join(base_path, str(n + i) + ".pkl"), "wb") as f:
                pickle.dump(acts, f)

    def _load_models(self):
        models = {(t, l, c): [] for t in self.targets for c in self.concept_names + self.control_concept_names for l in
                  self.layers}
        for p in os.listdir(self.get_path(self._CONF_CAV)):
            with tf.io.gfile.GFile(self.get_path([self._CONF_CAV, p]), 'rb') as pkl_file:
                save_dict = pickle.load(pkl_file)
            index = (save_dict["target"], save_dict["layer"], save_dict["concept"])
            models[index].append(save_dict["model"])
        return models

    def _load_cavs(self):
        models = {(t, l, c): [] for t in self.targets for c in self.concept_names + self.control_concept_names for l in
                  self.layers}
        for p in os.listdir(self.get_path(self._CONF_CAV)):
            with tf.io.gfile.GFile(self.get_path([self._CONF_CAV, p]), 'rb') as pkl_file:
                save_dict = pickle.load(pkl_file)
            if save_dict["target"] == "0" and save_dict["concept"] in self.concept_names:
                index = (save_dict["target"], save_dict["layer"], save_dict["concept"])
                models[index].append(CAV.load_cav(self.get_path([self._CONF_CAV, p])))
        return models

    def _load_derivatives(self):
        return {(k[0], k[1], i["cav_concept"]): np.array(i["val_directional_dirs"])
                for k, v in self.results.items() for i in v}

    def _compose_config(self):
        super(TCAVBaseMonitorExtractor, self)._compose_config()
        self.config["models"] = self._load_cavs()
        # self.config["derivatives"] = self._load_derivatives()

    def _preprocess(self):
        self.results = self.run_tcav(attach_header=False, clean_output=False)

    def _compute_relevant(self):
        relevant_dic = {(t,l,c): False for t in self.targets for l in self.layers for c in self.concept_names}
        results = []
        header_row = None
        for i, (k, v) in enumerate(self.config["tcavs"].items()):
            res = plot_results(v, num_random_exp=self.num_exp,
                                   random_concepts=self.control_concept_names, display=False)
            if i == 0:
                header_row = res[0]
            results += res[1:]
        df = pd.DataFrame(results, columns=header_row)
        df = df[df["significant"] == True]
        cutoff = find_cutoff(df["tcav"].tolist())
        df = df[df["tcav"] >= cutoff]
        for r in df.iterrows():
            row = r[1]
            relevant_dic[(row.at["target"], row.at["bottleneck"], row.at["concept"])] = True
        return relevant_dic

    def tcavs_as_table(self):
        results = []
        header_row = None
        for i, (k, v) in enumerate(self.config["tcavs"].items()):
            res = plot_results(v, num_random_exp=self.num_exp,
                               random_concepts=self.control_concept_names, display=False)
            if i == 0:
                header_row = res[0]
            results += res[1:]
        df = pd.DataFrame(results, columns=header_row)
        return df

    def _load_config(self, config=None):
        super(TCAVBaseMonitorExtractor, self)._load_config(config)
        # for k, v in self.config["tcavs"].items():
        #     for r in v:
        #         print(r.keys())
        self.relevant_elements = self._compute_relevant()

    def _extract(self, data, labels=None):
        pred, act = self.network.activations(data)
        if labels is None:
            results = [self._filter(self._extract_logic(data[i], pred[i], [a[i] for a in act[:-1]]))
                       for i in range(len(pred))]
        else:
            results = [self._filter(self._extract_logic(data[i], pred[i], [a[i] for a in act[:-1]], labels[i]), labels[i], pred[i])
                       for i in range(len(pred))]
        return pred, results

    def _extract_logic(self, data, predictions, activations, labels=None):
        pass

    def _filter(self, data, labels=None, prediction=None, *args, **kwargs):
        if data is None:
            return None
        if self.reasoner is None:
            return data
        else:
            print("-" * 100)
            print(self._parse_prediction(prediction), labels)
            print("-" * 100)
            res = self.reasoner.reason(data)

            return res

    def _aggregate_extract(self, results):
        if results is None:
            return None
        val = results.sum(axis=1) / (self.num_exp * results.shape[1])
        results = []
        for i in range(val.shape[1]):
            res = []
            for j in range(val.shape[0]):
                res.append((self.concept_names[j], val[j, i]))
            results.append(res)
        return results

    def _parse_prediction(self, prediction):
        if self.use_prediction:
            return [self.targets[np.argmax(prediction)]]
        else:
            return self.targets

    def _parse_layers(self, layer):
        return layer

    def _get_models(self, target, layer, concept):
        return [c.l_model for c in self.config["models"][(target, layer, concept)]]

    def _get_derivatives(self, target, layer, concept):
        return self.config["derivatives"][(target, layer, concept)]

    def _get_cavs(self, target, layer, concept):
        return self.config["models"][("0", layer, concept)]

    def _get_tcav_results(self):
        return self.config["tcav"]

    def _is_relevant(self, target, layer, concepts):
        return self.relevant_elements[(target, layer, concepts)]


class TCAVCavExtractor(TCAVBaseMonitorExtractor):
    _NAME = "cav_extractor"

    def _load_config(self, config=None):
        super(TCAVCavExtractor, self)._load_config(config)
        # self.config["models"] = self._load_cavs()
        # self._write_config()
        # sys.exit()
        self.concept_map = {c.get_name(): i for i, c in enumerate(self.concepts)}
        self.rev_concept_map = {i: c.get_name() for i, c in enumerate(self.concepts)}

    def _extract_logic(self, data, predictions, activations, labels=None):
        # print(labels)
        pred = self._parse_prediction(predictions)
        result = []
        for c in self.concept_names:
            res = []
            acc = []
            for l in self.layers:
                for p in pred:
                    if not self.only_relevant or self._is_relevant(p, l, c):
                        for lm in self._get_cavs(p, l, c):
                            res.append((lm.l_model.predict([np.reshape(activations[l], -1)])[0] + 1) % 2)
                            acc.append(lm.accuracies[c])
            result.append((c, np.average(res, weights=acc)))
        return result

        # for k, ml in self.config["models"].items():
        #     print(k)
        #     for m in ml:
        #         f_act = np.array([np.reshape(i, -1) for i in activations[k[1]]])
        #         res = (np.array(m.predict(f_act)) + 1) % 2
        #         result[self.concept_map[k[2]]][k[1]-1] += res

    # def _extract_logic(self, predictions, activations):
    #     result = np.zeros((len(self.concepts), len(self.layers), len(predictions)))
    #     print(predictions, activations)
    #     for k, ml in self.config["models"].items():
    #         print(k)
    #         for m in ml:
    #             f_act = np.array([np.reshape(i, -1) for i in activations[k[1]]])
    #             res = (np.array(m.predict(f_act)) + 1) % 2
    #             result[self.concept_map[k[2]]][k[1]-1] += res
    #     return result

    # def _voting2(self, results):
    #     val = results.sum(axis=1)
    #     val = val - val.max(axis=0)
    #     results = []
    #     for i in range(val.shape[1]):
    #         res = []
    #         for j in range(val.shape[0]):
    #             if val[j, i] == 0:
    #                 res.append(self.concept_names[j])
    #         results.append(res)
    #     return results
    #
    # def _voting(self, results):
    #     res_table = []
    #     for i in results:
    #         vals = np.sort(np.array([k[1] for k in i]))
    #         a = vals.reshape(-1, 1)
    #         kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(a)
    #         s = np.linspace(0, 1, num=100)
    #         e = kde.score_samples(s.reshape(-1, 1))
    #         mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    #         cutoff = s[mi[-1]]
    #         res = []
    #         for j in sorted(i, key=lambda x: x[1], reverse=True):
    #             if j[1] >= cutoff:
    #                 res.append(j)
    #         res_table.append(res)
    #         # plt.plot(s, e)
    #         # plt.show()
    #     return res_table


class TCAVSaliencyExtractor(TCAVCavExtractor):
    _NAME = "saliency_extractor"

    def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
                 num_exp=500, alpha=0.1, split=False, use_prediction=True, quantile=0.6, positive_only=True, only_relevant=False,
                 root="", reasoner=None, *args, **kwargs):
        super(TCAVSaliencyExtractor, self).__init__(network=network, dataset=dataset, concepts=concepts,
                                                    control_concepts=control_concepts, targets=targets, layers=layers,
                                                    max_examples=max_examples, num_exp=num_exp, alpha=alpha,
                                                    use_prediction=use_prediction, split=split, reasoner=reasoner,
                                                    only_relevant=only_relevant, root=root)
        self.quantile = quantile
        self.cutoff = None
        self.positive_only = positive_only
        self.cav_reasoner = None

    def _extract(self, data, labels=None):
        pred, act = self.network.activations(data)
        if labels is None:
            results = [self._process_datapoint(data[i], pred[i], [a[i] for a in act[:-1]])
                       for i in range(len(pred))]
        else:
            results = [self._process_datapoint(data[i], pred[i], [a[i] for a in act[:-1]], labels[i], pred[i])
                       for i in range(len(pred))]
        return pred, results

    def _process_datapoint(self, data, prediction, activation, *args, **kwargs):
        results_cav = self._filter(self._extract_logic(data, prediction, activation), *args, **kwargs)
        concepts = []
        all_results = []
        if len(results_cav)!=0:
            concepts = self.reasoner.get_low_level_concepts(results_cav)
            # print(concepts)
            # print(self.concept_map)
            results_dir, all_results = self._extract_logic_dir(data, prediction, activation, concepts)
            # min_vals = np.min(results_dir, axis=1)
            # results_dir[min_vals < 0] = results_dir[min_vals < 0] + np.abs(min_vals[min_vals < 0])[:, None] # results_dir[np.max(results_dir, axis=1) > 0]
            # print(results_dir)
            if len(results_dir)>0:
                # print(results_dir)
                # results = np.mean(results_dir[:, 0] - results_dir[:, 1])
                results = np.mean(results_dir, axis=0)
            else:
                results = [0,0]

            # if results > -0.1 and results < 0.1:
            #     print("{0} -> Bias: Undefined".format(round(results, 5)))
            # elif results >= 0:
            #     print("{0} -> Bias: Shapes".format(round(results, 5)))
            # elif results < 0:
            #     print("{0} -> Bias: Color".format(round(results, 5)))
        else:
            results = [0,0]
        try:
            print(concepts ,results, "->", concepts[np.argmax(results)])
        except Exception:
            pass
        res = [results_cav, results, all_results, concepts, self.concept_names]
        # print(res)
        return res

    def _extract_logic_dir(self, data, predictions, activations, concepts=None):
        pred = self._parse_prediction(predictions)
        #concepts = self.concept_names # if concepts is None else concepts
        results = []
        df = []
        for l in self.layers:
            row = []
            for c in self.concept_names:
                act = activations[self._parse_layers(l)]
                s = []
                for p in pred:
                    deriv = [TCAV.get_directional_dir(self.mymodel, p, c, cav, [act], [data])[0] for cav in self._get_cavs(p, l, c)]
                    s += deriv
                    for i in deriv:
                        df.append([p, l, c, i])
                    acc = [lm.accuracies[c] for lm in self._get_cavs(p,l,c)]
                row.append(np.average(s, weights=acc))
            results.append(row)

        results = np.array(results)
        all_results = results.copy()
        # print(results)
        results = results * -1
        # results[results < 0] = 0
        results = results / np.abs(np.max(results, axis=1)[:, None] - np.min(results, axis=1)[:, None])
        # print(results)
        # results = results.mean(axis=0)
        # results = [(c, results[i]) for i, c in enumerate(self.concept_names)]
        if concepts is not None:
            # print([self.concept_map[c] for c in concepts])
            # print(results[:, [self.concept_map[c] for c in concepts]])
            # print(sorted([self.concept_map[c] for c in concepts]))
            # print(results[:, sorted([self.concept_map[c] for c in concepts])])
            results = results[:, [self.concept_map[c] for c in concepts]]
            # print()
        return results, all_results


class TCAVSaliencyExtractor2(TCAVBaseMonitorExtractor):
    _NAME = "saliency_extractor2"

    def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
                 num_exp=500, alpha=0.1, split=False, use_prediction=True, quantile=0.6, positive_only=True, only_relevant=False,
                 root="", reasoner=None, *args, **kwargs):
        super(TCAVSaliencyExtractor2, self).__init__(network=network, dataset=dataset, concepts=concepts,
                                                    control_concepts=control_concepts, targets=targets, layers=layers,
                                                    max_examples=max_examples, num_exp=num_exp, alpha=alpha,
                                                    use_prediction=use_prediction, split=split, reasoner=reasoner,
                                                    only_relevant=only_relevant, root=root)
        self.quantile = quantile
        self.cutoff = None
        self.positive_only = positive_only


    def _load_config(self, config=None):
        super(TCAVSaliencyExtractor2, self)._load_config(config)
        self.cutoff = self._compute_cutoff()

    def _compute_cutoff3(self):
        res_dic = {}
        for t in self.targets:
            for l in self.layers:
                deriv = []
                for c in self.concept_names:
                    # print(self.config["derivatives"][(t, l, c.get_name())])
                    deriv += list(self.config["derivatives"][(t, l, c)])
                # sns.displot(deriv, bins=30)
                # plt.show()
                for c in self.concept_names:
                    res_dic[(t,l,c)] = np.quantile(deriv, self.quantile)
        return res_dic

    def _compute_cutoff(self):
        res_dic = {}

        for l in self.layers:
            deriv = []
            for t in self.targets:
                for c in self.concept_names:
                    # print(self.config["derivatives"][(t, l, c.get_name())])
                    if self._is_relevant(t, l, c):
                        deriv += list(self.config["derivatives"][(t, l, c)])
            # sns.displot(deriv, bins=30)
                # plt.show()
            for c in self.concept_names:
                for t in self.targets:
                    #print( np.max(deriv))
                    res_dic[(t,l,c)] = np.max(deriv)
                    # res_dic[(t, l, c)] = np.quantile(deriv, self.quantile)
        return res_dic

    def _get_cutoff(self, target, layer, concept):
        return self.cutoff[target,layer,concept]

    def _compute_cutoff2(self):
        if self.positive_only:
            return {(t, l, c.get_name()):
                        np.quantile([d for d in self.config["derivatives"][(t, l, c.get_name())] if d > 0],
                                    self.quantile)
                    for t in self.targets for l in self.layers for c in self.concepts}
        else:
            return {(t, l, c.get_name()):
                        np.quantile(self.config["derivatives"][(t, l, c.get_name())], self.quantile)
                    for t in self.targets for l in self.layers for c in self.concepts}

    def _extract_logic(self, data, predictions, activations, labels=None):
        pred = self._parse_prediction(predictions)
        results = []
        df = []
        for l in self.layers:
            row = []
            for c in self.concept_names:
                act = activations[self._parse_layers(l)]
                s = []
                for p in pred:
                    deriv = [TCAV.get_directional_dir(self.mymodel, p, c, cav, [act], [data])[0] for cav in self._get_cavs(p, l, c)]
                    s += deriv
                    for i in deriv:
                        df.append([p, l, c, i])
                row.append(np.average(s))
                #weights=[cav.accuracies[c] for p in pred for cav in self._get_cavs(p, l, c)]
            results.append(row)
        df = pd.DataFrame(df, columns=["t", "l", "c", "i"])
        df.to_csv("results/data_b_{0}.csv".format(len(os.listdir("results"))))
        results = np.array(results)
        results = results *-1
        absval = np.abs(results)
        # results[results < 0] = 0
        results = results / np.max(absval, axis=1)[:, None]
        results = results.mean(axis=0)
        results = [(c, results[i]) for i, c in enumerate(self.concept_names)]
        return results

    def _extract_logic2(self, data, predictions, activations, labels=None):
        pred = self._parse_prediction(predictions)
        results = []
        for l in self.layers:
            for c in self.concept_names:
                act = activations[self._parse_layers(l)]
                for p in pred:
                    if not self.only_relevant or self._is_relevant(p, l, c):
                        for cav in self._get_cavs(p, l, c):
                            s = TCAV.get_directional_dir(self.mymodel, p, c, cav, [act], [data])[0]
                            results.append([p, l, c, s])
        results = self._clean_derivatives2(results)
        return results

    def _clean_derivatives2(self, data):
        df = pd.DataFrame(data, columns=["t", "l", "c", "s"])
        df = df[df["s"] > 0]
        for i in range(0, 7):
            df.loc[df["l"] == i, "s"] = df[df["l"] == i]["s"] / df[df["l"] == i]["s"].max()
#
# class TCAVSaliencyExtractor2(TCAVBaseMonitorExtractor):
#
#     _NAME = "saliency_extractor"
#
#     def __init__(self, network, dataset, concepts, control_concepts, targets=None, layers=None, max_examples=500,
#                  num_exp=500, alpha=0.1, split=False, quantile=0.9, root="", *args, **kwargs):
#         super(TCAVSaliencyExtractor2, self).__init__(network=network, dataset=dataset, concepts=concepts,
#                                                     control_concepts=control_concepts, targets=targets, layers=layers,
#                                                     max_examples=max_examples,num_exp=num_exp, alpha=alpha,
#                                                     split=split, root=root)
#         self.quantile = quantile
#         self.cutoff = None
#
#     def _compose_config(self):
#         super(TCAVSaliencyExtractor2, self)._compose_config()
#         self.config["derivatives"] = self.results
#         if isinstance(self.quantile, float):
#             self.config["quantile"] = {(t, l, c.get_name()): self.quantile for t in self.targets for l in self.layers for c in
#                                        self.concepts}
#         else:
#             self.config["quantile"] = self.quantile
#         self.config["cavs"] = self._load_cavs()
#
#     def _load_config(self):
#         super(TCAVSaliencyExtractor2, self)._load_config()
#         print(self.config)
#         self.quantile = self.config["quantile"]
#         self.cavs = self.config["cavs"]
#         self.cutoff = {(t, l, c.get_name()): 0 for t in self.targets for l in self.layers for c in self.concepts}
#         for k, v in self.config["derivatives"].items():
#             print(k)
#             if k[2] in self.concept_names:
#                 self.cutoff[k] = np.quantile(v, self.quantile[k])
#
#     def _extract(self, data):
#         pred, act = self.network.activations(data)
#         resutls = []
#         for k, m in self.config["cavs"].items():
#             res = []
#             for i in range(len(pred)):
#                 target = self.targets[np.argmax(pred[i])]
#                 salience = TCAV.get_directional_dir(self.mymodel, target, k[1], m,
#                                                     [act[k[0]][i]], [data[i]])
#                 print(k, salience, self.cutoff[target, k[0], k[1]])
#                 res.append()
#         return [], []
#
#     def _load_cavs(self):
#         cavs = {}
#         for p in os.listdir(self.get_path(self._CONF_CAV)):
#             cav = CAV.load_cav(self.get_path([self._CONF_CAV, p]))
#             cavs[cav.bottleneck, cav.concepts[0]] = cav
#         return cavs
#
#     def _voting(self, results):
#         val = results.sum(axis=1)
#         val = val - val.max(axis=0)
#         results = []
#         for i in range(val.shape[1]):
#             res = []
#             for j in range(val.shape[0]):
#                 if val[j, i] == 0:
#                     res.append(self.concept_names[j])
#             results.append(res)
#         return results
#
#
