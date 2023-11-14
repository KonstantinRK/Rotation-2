import pipelines as p_pipelines
import monitors as p_monitors
import networks as p_networks
import datasets as p_datasets
from utils import BaseClass
import shutil
import os
import reasoner as p_reason


class Experiment(BaseClass):
    _NAME = "exp"

    def __init__(self, root="", *args, **kwargs):
        super(Experiment, self).__init__(root, *args, **kwargs)
        self.network = None
        self.dataset = None
        self.datasets = []
        self.monitors = None
        self.pipelines = None
        self.train_pipeline = None
        self.concepts = None
        self.control_concepts = None

    def _define_experiment(self, dataset_root, network_root=None):
        pass

    def initialise(self, dataset_root="", network_root=""):
        self._define_experiment(dataset_root, network_root)

    def preprocess(self):
        [d.create() for d in self.datasets]
        self._additional_preprocessing()
        self.network.classes = len(self.dataset.get_labels())
        self.train_pipeline.preprocess()
        [p.preprocess() for p in self.pipelines]

    def setup(self):
        self._setup()
        self.train_pipeline.setup()
        self.network.setup()
        [d.setup() for d in self.datasets]
        self._additional_setup()
        [c.setup() for c in self.pipelines]

    def run(self, overwrite=False):
        if isinstance(overwrite, bool):
            overwrite = [overwrite] * len(self.pipelines)
        for i, p in enumerate(self.pipelines):
            print("")
            print("")
            print("")
            print("-"*200)
            print("RUN:", p.get_name(), "({})".format(str(i)))
            print("-" * 200)
            print("")
            print("")
            print("")

            p.load()
            print(p.network.model.summary())
            p.run(overwrite[i])

    def analyse(self):
        pass

    def export(self, path):
        for p in self.pipelines:
            p.load()
            p.export(path)

    def get_result_path(self):
        return self.pipelines.get_result_path()

    def get_results(self):
        return {p.get_name(): p.get_results() for p in self.pipelines}

    def _additional_setup(self):
        pass

    def _additional_preprocessing(self):
        pass

    def reset(self, datasets=False):
        self.network.delete()
        for p in self.pipelines:
            p.delete()
        self.delete()
        if datasets:
            self.dataset.delete()

    def clean(self):
        for p in self.pipelines:
            p.clean()

    def export_results(self, path):
        path = os.path.join(path, self.get_name())
        os.makedirs(path, exist_ok=True)
        for k, v in self.get_results().items():
            v.to_csv(os.path.join(path, k + ".csv"))


class TcavShapes(Experiment):
    _NAME = "tcav_shapes"

    def _define_experiment(self, dataset_root, network_root=None):
        if network_root is None:
            network_root = self.get_path()
        inp_size = (32, 32)
        self._define_dataset(dataset_root, inp_size)
        self._define_network(network_root, inp_size)
        self._define_concepts(dataset_root, inp_size)
        self._define_control_concepts(dataset_root, inp_size)
        self._define_training_pipeline()
        self._define_pipelines()

    def _additional_setup(self):
        [c.setup() for c in self.concepts]
        [c.setup() for c in self.control_concepts]

    def _additional_preprocessing(self):
        [c.create() for c in self.concepts]
        [c.create() for c in self.control_concepts]

    def _define_network(self, network_root, inp_size):
        self.network = p_networks.LeNet5(inp_size=(*inp_size, 3), root=network_root,
                                         loss="sparse_categorical_crossentropy")

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.Shapes(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]

    def _define_training_pipeline(self):
        self.train_pipeline = p_pipelines.TrainPipeLine(network=self.network, dataset=self.dataset,
                                                        root=self.get_path(), epochs=50)

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.ModTCAVPipeLine(network=self.network,
                                        dataset=self.dataset,
                                        concepts=self.concepts,
                                        control_concepts=self.control_concepts,
                                        layers=None, targets=None,
                                        root=self.get_path()),
            # p_pipelines.ModRelTCAVPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                concepts=self.concepts,
            #                                control_concepts=self.control_concepts,
            #                                layers=None, targets=None,
            #                                root=self.get_path()),
            p_pipelines.TCAVPipeLine(network=self.network,
                                     dataset=self.dataset,
                                     concepts=self.concepts,
                                     control_concepts=self.control_concepts,
                                     layers=None, targets=None,
                                     root=self.get_path()),
            # p_pipelines.RelTCAVPipeLine(network=self.network,
            #                             dataset=self.dataset,
            #                             concepts=self.concepts,
            #                             control_concepts=self.control_concepts,
            #                             layers=None, targets=None,
            #                             root=self.get_path()),
        ]

    @staticmethod
    def _init_concepts(dataset_root, inp_size):
        return [p_datasets.Circle(root=dataset_root, size=inp_size),
                p_datasets.Rectangle(root=dataset_root, size=inp_size),
                p_datasets.Triangle(root=dataset_root, size=inp_size),
                p_datasets.Red(root=dataset_root, size=inp_size),
                p_datasets.Blue(root=dataset_root, size=inp_size),
                p_datasets.Green(root=dataset_root, size=inp_size),
                p_datasets.RedCircle(root=dataset_root, size=inp_size),
                p_datasets.RedRectangle(root=dataset_root, size=inp_size),
                p_datasets.RedTriangle(root=dataset_root, size=inp_size),
                p_datasets.BlueCircle(root=dataset_root, size=inp_size),
                p_datasets.BlueRectangle(root=dataset_root, size=inp_size),
                p_datasets.BlueTriangle(root=dataset_root, size=inp_size),
                p_datasets.GreenCircle(root=dataset_root, size=inp_size),
                p_datasets.GreenRectangle(root=dataset_root, size=inp_size),
                p_datasets.GreenTriangle(root=dataset_root, size=inp_size)
                ]

    def _define_concepts(self, dataset_root, inp_size):
        self.concepts = self._init_concepts(dataset_root, inp_size)

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 p_datasets.ShapesRan2(root=dataset_root, size=inp_size),
                                 p_datasets.ShapesRan3(root=dataset_root, size=inp_size),
                                 ]

    def reset(self, datasets=False):
        super(TcavShapes, self).reset(datasets)
        if datasets:
            for c in self.concepts:
                c.delete()
            for c in self.control_concepts:
                c.delete()


# class TcavShapesRel(TcavShapes):
#     _NAME = "tcav_shapes_rel"
#
#     def _define_control_concepts(self, dataset_root, inp_size):
#         self.control_concepts = self._init_concepts(dataset_root, inp_size)
#

class TcavShapesControl(TcavShapes):
    _NAME = "tcav_shapes_control"

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.ShapesControll(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]


class TcavShapesTest(TcavShapes):
    _NAME = "tcav_shapes_test"

    @staticmethod
    def _init_concepts(dataset_root, inp_size):
        return [p_datasets.Red(root=dataset_root, size=inp_size),
                p_datasets.Blue(root=dataset_root, size=inp_size),
                p_datasets.Green(root=dataset_root, size=inp_size),
                ]

    def _define_pipelines(self):
        self.pipelines = [
            # p_pipelines.ModTCAVPipeLine(network=self.network,
            #                             dataset=self.dataset,
            #                             concepts=self.concepts,
            #                             control_concepts=self.control_concepts,
            #                             layers=None, targets=None,
            #                             root=self.get_path()),
            # p_pipelines.ModRelTCAVPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                concepts=self.concepts,
            #                                control_concepts=self.control_concepts,
            #                                layers=None, targets=None,
            #                                root=self.get_path()),
            p_pipelines.TCAVPipeLine(network=self.network,
                                     dataset=self.dataset,
                                     concepts=self.concepts,
                                     control_concepts=self.control_concepts,
                                     layers=None, targets=None,
                                     root=self.get_path()),
            # p_pipelines.RelTCAVPipeLine(network=self.network,
            #                             dataset=self.dataset,
            #                             concepts=self.concepts,
            #                             control_concepts=self.control_concepts,
            #                             layers=None, targets=None,
            #                             root=self.get_path()),
        ]


class TcavShapesExtended(TcavShapes):
    _NAME = "tcav_shapes_ext"

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 p_datasets.ShapesRan2(root=dataset_root, size=inp_size),
                                 p_datasets.ShapesRan3(root=dataset_root, size=inp_size),
                                 p_datasets.RCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.BCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.GCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.CIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.CMNISTR1(root=dataset_root, size=inp_size),
                                 p_datasets.CMNISTR2(root=dataset_root, size=inp_size),
                                 p_datasets.CMNISTR3(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST1(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST2(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST3(root=dataset_root, size=inp_size),
                                 ]


class TcavShapesExtended2(TcavShapes):
    _NAME = "tcav_shapes_ext2"

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 p_datasets.RCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.BCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.GCIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.CIFAR10(root=dataset_root, size=inp_size),
                                 p_datasets.CMNISTR3(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST1(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST2(root=dataset_root, size=inp_size),
                                 p_datasets.CMNIST3(root=dataset_root, size=inp_size),
                                 ]


class TcavShapesExtControl(TcavShapesExtended):
    _NAME = "tcav_shapes_ext_control"

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.ShapesControll(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]


class TcavCavTestMonitoring(TcavShapesTest):
    _NAME = "tcav_test_monitoring_shapes"

    def _define_dataset(self, dataset_root, inp_size):
        super(TcavCavTestMonitoring, self)._define_dataset(dataset_root, inp_size)
        self.monitor_dataset = p_datasets.ShapesLabeled(root=dataset_root, size=inp_size)
        self.datasets.append(self.monitor_dataset)

    @staticmethod
    def validation_foo(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        if len(set(results).difference(set(labels + [labels[1] + "_" + labels[0]]))) > 0:
            return 0
        vote = 0
        if labels[0] in results:
            vote += 0.5
        if labels[1] in results:
            vote += 0.5
        if labels[1] + "_" + labels[0] in results:
            vote = 1
        return vote

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 ]

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.SaliencyMonitorPipeLine(network=self.network,
                                                dataset=self.dataset,
                                                monitor_dataset=self.monitor_dataset,
                                                concepts=self.concepts,
                                                validation_foo=self.validation_foo,
                                                control_concepts=self.control_concepts,
                                                layers=[2, 3], targets=None, split=True, num_exp=2,
                                                root=self.get_path()),
        ]


class TcavCavTest2Monitoring(TcavCavTestMonitoring):
    _NAME = "tcav_test2_monitoring_shapes"

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 p_datasets.ShapesRan2(root=dataset_root, size=inp_size),
                                 ]

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.TCAVPipeLine(network=self.network,
                                     dataset=self.dataset,
                                     monitor_dataset=self.monitor_dataset,
                                     concepts=self.concepts,
                                     validation_foo=self.validation_foo,
                                     control_concepts=self.control_concepts,
                                     layers=None, targets=None, split=True, num_exp=100,
                                     root=self.get_path()),
        ]


class TcavCavMonitoring(TcavShapesExtended2):
    _NAME = "tcav_monitoring_shapes"

    def _define_control_concepts(self, dataset_root, inp_size):
        self.control_concepts = [p_datasets.ShapesRan1(root=dataset_root, size=inp_size),
                                 ]

    def _define_dataset(self, dataset_root, inp_size):
        super(TcavCavMonitoring, self)._define_dataset(dataset_root, inp_size)
        self.monitor_dataset = p_datasets.ShapesLabeled(root=dataset_root, size=inp_size)
        self.datasets.append(self.monitor_dataset)

    def filter_foo(self, results):
        inp_vals = []
        for r in results:
            if self.type_map[r[0]] == "color":
                inp_vals.append((r[0], None, r[1]))
            elif self.type_map[r[0]] == "shape":
                inp_vals.append((None, r[0], r[1]))
            elif self.type_map[r[0]] == "composite":
                v = r[0].split("_")
                inp_vals.append((v[0], v[1], r[1]))
                inp_vals.append((None, v[1], r[1]))
                inp_vals.append((v[0], None, r[1]))
        res = []
        for r1 in inp_vals:
            add = True
            for r2 in inp_vals:
                if type(r1[0]) == type(r2[0]) and type(r1[1]) == type(r2[1]) and r1[2] < r2[2]:
                    add = False
            if add:
                res.append(r1)
        count = [0, 0]
        for r in res:
            if r[0] is not None:
                count[0] += 1
            if r[1] is not None:
                count[1] += 1
        if count[0] == 1 and count[1] == 1 and len(res) == 2:
            c = None
            s = None
            for r in res:
                if r[0] is None:
                    s = r
                elif r[1] is None:
                    c = r
            res.append((c[0], s[1], min(c[2], s[2])))
        # print(res_list)
        return res

    # def filter_foo(self, results):
    #     results = [r for r in results if self.type_map[r[0]] != "composite"]
    #     value_map = {k[0]: k[1] for k in results}
    #     concept_set = []
    #     for r1 in results:
    #         add = True
    #         for r2 in results:
    #             if self.type_map[r1[0]] == self.type_map[r2[0]] and value_map[r1[0]] < value_map[r2[0]]:
    #                 add = False
    #         if add:
    #             concept_set.append(r1[0])
    #     if len(concept_set) == 2:
    #         b_color = None
    #         b_shape = None
    #         for c in concept_set:
    #             if self.type_map[c] == "color":
    #                 b_color = c
    #             if self.type_map[c] == "shape":
    #                 b_shape = c
    #         if b_shape is not None and b_color is not None:
    #             concept_set.append(b_color + "_" + b_shape)

    @staticmethod
    def validation_foo(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        labels = set(labels + [labels[1] + "_" + labels[0]])
        return len(results.difference(labels)) == 0

    @staticmethod
    def validation_foo2(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        labels = set(labels + [labels[1] + "_" + labels[0]])
        res_list = []
        # print(res)
        for r in results:
            if r[0] is not None and r[1] is not None:
                res_list.append(r[0] + "_" + r[1])
            elif r[0] is not None:
                res_list.append(r[0])
            elif r[1] is not None:
                res_list.append(r[1])
        results = set(res_list)
        if len(results.difference(labels)) > 0:
            return 0
        else:
            return len(results.intersection(labels)) / len(labels)

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.CavMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner5(),
                                           layers=None, targets=None, split=True, num_exp=50,
                                           root=self.get_path()),
        ]


class TcavBaseCavMonitoring(TcavCavMonitoring):
    _NAME = "tcav_base_monitoring_shapes"

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.BaseMonitorPipeLine(network=self.network,
                                            dataset=self.dataset,
                                            monitor_dataset=self.monitor_dataset,
                                            concepts=self.concepts,
                                            validation_foo=self.validation_foo,
                                            control_concepts=self.control_concepts,
                                            reasoner=p_reason.ShapeConceptReasoner3(),
                                            layers=None, targets=None, split=True, num_exp=25,
                                            root=self.get_path()),
        ]


class TcavBaseCavMonitoringControl(TcavBaseCavMonitoring):
    _NAME = "tcav_base_monitoring_shapes_control"

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.ShapesControll(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]
        self.monitor_dataset = p_datasets.ShapesLabeled(root=dataset_root, size=inp_size)
        self.datasets.append(self.monitor_dataset)

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.BaseMonitorPipeLine(network=self.network,
                                            dataset=self.dataset,
                                            monitor_dataset=self.monitor_dataset,
                                            concepts=self.concepts,
                                            validation_foo=self.validation_foo,
                                            control_concepts=self.control_concepts,
                                            reasoner=p_reason.ShapeConceptReasoner3(),
                                            layers=None, targets=None, split=True, num_exp=100,
                                            root=self.get_path()),
        ]


class TcavFinCavMonitoringControl(TcavBaseCavMonitoring):
    _NAME = "tcav_final_monitoring_shapes_control"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner6()


    def _define_training_pipeline(self):
        self.train_pipeline = p_pipelines.TrainPipeLine(network=self.network, dataset=self.dataset,
                                                        root=self.get_path(), epochs=100)

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.ShapesControll(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]
        self.monitor_dataset = p_datasets.ShapesLabeled(root=dataset_root, size=inp_size)
        self.datasets.append(self.monitor_dataset)

    @staticmethod
    def validation_foo2(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        return 1

    @staticmethod
    def validation_foo(predictions, results, labels):
        labels = set(labels + [labels[1] + "_" + labels[0]])
        # print(results, labels)
        return len(set(results).intersection(labels)) / len(labels)

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.SaliencyMonitorPipeLine(network=self.network,
                                                dataset=self.dataset,
                                                monitor_dataset=self.monitor_dataset,
                                                concepts=self.concepts,
                                                validation_foo=self.validation_foo2,
                                                control_concepts=self.control_concepts,
                                                reasoner=self.get_reasoner(),
                                                layers=None, targets=None, split=True, num_exp=75,
                                                batches=50, batch_size=50,
                                                root=self.get_path()),

            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=self.get_reasoner(),
            #                                layers=None, targets=None, split=True, num_exp=75,
            #                                batches=100, batch_size=50,
            #                                root=self.get_path()),

        ]

    # def _define_pipelines(self):
    #     self.pipelines = [
    #         p_pipelines.SaliencyMonitorPipeLine(network=self.network,
    #                                         dataset=self.dataset,
    #                                         monitor_dataset=self.monitor_dataset,
    #                                         concepts=self.concepts,
    #                                         validation_foo=self.validation_foo,
    #                                         control_concepts=self.control_concepts,
    #                                         reasoner=p_reason.ShapeConceptReasoner3(),
    #                                         layers=None, targets=None, split=True, num_exp=75,
    #                                         root=self.get_path()),
    #     ]


class TcavFinCavMonitoringControl2(TcavFinCavMonitoringControl):
    _NAME = "tcav_final_monitoring_shapes_control2"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner4()


class TcavFinCavMonitoringControl3(TcavFinCavMonitoringControl):
    _NAME = "tcav_final_monitoring_shapes_control3"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner5()


class TcavFinCavMonitoringControl4(TcavFinCavMonitoringControl):
    _NAME = "tcav_final_monitoring_shapes_control4"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner7()



class TcavFinCavMonitoring(TcavBaseCavMonitoring):
    _NAME = "tcav_final_monitoring_shapes"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner6()

    def _define_training_pipeline(self):
        self.train_pipeline = p_pipelines.TrainPipeLine(network=self.network, dataset=self.dataset,
                                                        root=self.get_path(), epochs=100)

    def _define_dataset(self, dataset_root, inp_size):
        self.dataset = p_datasets.Shapes2(root=dataset_root, size=inp_size)
        self.datasets = [self.dataset]
        self.monitor_dataset = p_datasets.ShapesLabeled(root=dataset_root, size=inp_size)
        self.datasets.append(self.monitor_dataset)

    @staticmethod
    def validation_foo2(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        return 1

    @staticmethod
    def validation_foo(predictions, results, labels):
        labels = set(labels + [labels[1] + "_" + labels[0]])
        # print(results, labels)
        return len(set(results).intersection(labels))/len(labels)

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.SaliencyMonitorPipeLine(network=self.network,
                                                dataset=self.dataset,
                                                monitor_dataset=self.monitor_dataset,
                                                concepts=self.concepts,
                                                validation_foo=self.validation_foo2,
                                                control_concepts=self.control_concepts,
                                                reasoner=self.get_reasoner(),
                                                layers=None, targets=None, split=True, num_exp=75,
                                                batches=50, batch_size=50,
                                                root=self.get_path()),

            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=self.get_reasoner(),
            #                                layers=None, targets=None, split=True, num_exp=75,
            #                                batches=100, batch_size=50,
            #                                root=self.get_path()),


        ]


class TcavFinCavMonitoring2(TcavFinCavMonitoring):
    _NAME = "tcav_final_monitoring_shapes2"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner4()


class TcavFinCavMonitoring3(TcavFinCavMonitoring):
    _NAME = "tcav_final_monitoring_shapes3"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner5()


class TcavFinCavMonitoring4(TcavFinCavMonitoring):
    _NAME = "tcav_final_monitoring_shapes4"

    def get_reasoner(self):
        return p_reason.ShapeConceptReasoner7()


class TcavBaseCavMonitoringTest(TcavCavMonitoring):
    _NAME = "tcav_base_monitoring_shapes_test"

    def _define_pipelines(self):
        self.pipelines = [
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner1(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner2(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner3(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner4(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner5(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.CavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner6(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            # p_pipelines.ACavMonitorPipeLine(network=self.network,
            #                                 dataset=self.dataset,
            #                                 monitor_dataset=self.monitor_dataset,
            #                                 concepts=self.concepts,
            #                                 validation_foo=self.validation_foo,
            #                                 control_concepts=self.control_concepts,
            #                                 reasoner=p_reason.ShapeConceptReasoner1(),
            #                                 layers=None, targets=None, split=True, num_exp=25,
            #                                 batches=1000, batch_size=50,
            #                                 root=self.get_path()),
            # p_pipelines.ACavMonitorPipeLine(network=self.network,
            #                                dataset=self.dataset,
            #                                monitor_dataset=self.monitor_dataset,
            #                                concepts=self.concepts,
            #                                validation_foo=self.validation_foo,
            #                                control_concepts=self.control_concepts,
            #                                reasoner=p_reason.ShapeConceptReasoner2(),
            #                                layers=None, targets=None, split=True, num_exp=25,
            #                                batches=1000, batch_size=50,
            #                                root=self.get_path()),
            p_pipelines.ACavMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner3(),
                                           layers=None, targets=None, split=True, num_exp=25,
                                           batches=1000, batch_size=50,
                                           root=self.get_path()),
            p_pipelines.ACavMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner4(),
                                           layers=None, targets=None, split=True, num_exp=25,
                                           batches=1000, batch_size=50,
                                           root=self.get_path()),
            p_pipelines.ACavMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner5(),
                                           layers=None, targets=None, split=True, num_exp=25,
                                           batches=1000, batch_size=50,
                                           root=self.get_path()),
            p_pipelines.ACavMonitorPipeLine(network=self.network,
                                            dataset=self.dataset,
                                            monitor_dataset=self.monitor_dataset,
                                            concepts=self.concepts,
                                            validation_foo=self.validation_foo,
                                            control_concepts=self.control_concepts,
                                            reasoner=p_reason.ShapeConceptReasoner6(),
                                            layers=None, targets=None, split=True, num_exp=25,
                                            batches=1000, batch_size=50,
                                            root=self.get_path()),


        ]


class TcavBaseCavMonitoringTest2(TcavCavMonitoring):
    _NAME = "tcav_base_monitoring_shapes_test"

    @staticmethod
    def validation_foo(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        color = labels[1]
        shape = labels[0]
        # if predictions == 2:
        #     return results[1]
        return 1

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.SaliencyMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner3(),
                                           layers=None, targets=None, split=True, num_exp=25,
                                           batches=10, batch_size=50,
                                           root=self.get_path()),
        ]


class TcavBaseCavMonitoringControlTest(TcavBaseCavMonitoringControl):
    _NAME = "tcav_base_monitoring_shapes_control_test"

    @staticmethod
    def validation_foo(predictions, results, labels):
        # labels = set(labels + [labels[1] + "-" + labels[0]])
        return results[1] > -0.1

    def _define_pipelines(self):
        self.pipelines = [
            p_pipelines.SaliencyMonitorPipeLine(network=self.network,
                                           dataset=self.dataset,
                                           monitor_dataset=self.monitor_dataset,
                                           concepts=self.concepts,
                                           validation_foo=self.validation_foo,
                                           control_concepts=self.control_concepts,
                                           reasoner=p_reason.ShapeConceptReasoner3(),
                                           layers=None, targets=None, split=True, num_exp=25,
                                           batches=10, batch_size=50,
                                           root=self.get_path()),
        ]

