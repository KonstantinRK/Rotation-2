import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import BaseClass
import utils
import cv2
import sys
import random


class DataSet(BaseClass):

    _NAME = "ds"
    _TEMP_NAME = "temp"
    _TEST_NAME = "test"
    _TRAIN_NAME = "training"
    _VAL_NAME = "validation"

    def __init__(self, root="", overwrite=False, shuffle=True, seed=123, batch_size=30,
                 validation_split=0.1, *args, **kwargs):
        super(DataSet, self).__init__(root, *args, **kwargs)
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.overwrite = overwrite
        self._initial_processing()

    def _initial_processing(self):
        pass

    def translate_label(self, label):
        return str(label)

    def _manage_temp(self, init=True):
        if os.path.exists(self.get_path(self._TEMP_NAME)):
            shutil.rmtree(self.get_path(self._TEMP_NAME))
        if init:
            os.makedirs(self.get_path(self._TEMP_NAME))

    def get_path(self, file_name=None):
        if file_name is None:
            return os.path.join(self.root_dir, self.get_name())
        else:
            return os.path.join(self.root_dir, self.get_name(), file_name)

    def setup_dir(self):
        if self.overwrite and not os.path.exists(self.get_path()):
            os.makedirs(self.get_path())
        os.makedirs(self.get_path(), exist_ok=True)

    def _val_exists(self):
        return os.path.exists(self.get_path(self._VAL_NAME))

    def _create(self):
        pass

    def create(self):
        if self.overwrite or (not os.path.exists(self.get_path())) or self._TRAIN_NAME not in os.listdir(self.get_path()):
            self._create()

    def _create_from_dir(self, path, *args, **kwargs):
        self.setup_dir()
        self.import_data(path, *args, **kwargs)

    def import_data(self, path, copy=True):
        if copy:
            f = lambda x, y: shutil.copytree(x, y)
        else:
            f = lambda x, y: os.rename(x, y)
        f(os.path.join(path, self._TEST_NAME), self.get_path(self._TEST_NAME))
        f(os.path.join(path, self._TRAIN_NAME), self.get_path(self._TRAIN_NAME))
        try:
            f(os.path.join(path, self._VAL_NAME), self.get_path(self._VAL_NAME))
        except FileNotFoundError:
            pass

    def _load_test(self, with_labels=False, *args, **kwargs):
        pass

    def _load_train(self, with_labels=True, *args, **kwargs):
        pass

    def _load_val(self, with_labels=True, *args, **kwargs):
        pass

    def _load_size(self, load_foo, batch_size=None, batches=None, with_labels=None, *args, **kwargs):
        if batch_size is None:
            res = load_foo(with_labels=with_labels, *args, **kwargs)
        else:
            bs = self.batch_size
            self.batch_size = batch_size
            res = load_foo(with_labels=with_labels, *args, **kwargs)
            self.batch_size = bs
        if batches is None:
            return res
        else:
            return res

    # def _load_size(self, load_foo, batch_size=None, batches=None, with_labels=None, *args, **kwargs):
    #     print("A")
    #     if batch_size is None:
    #         return load_foo(with_labels=with_labels, *args, **kwargs)
    #     else:
    #         bs = self.batch_size
    #         self.batch_size = batch_size
    #         res = load_foo(batch_size=batch_size, batches=batches, with_labels=with_labels, *args, **kwargs)
    #         self.batch_size = bs
    #         print(batches)
    #         if batches is None:
    #             print("A")
    #             return next(res)
    #         else:
    #             for n in range(batches):
    #                 print("A")
    #                 yield next(res)

    def load_test(self, with_labels=False, batch_size=None, batches=None, *args, **kwargs):
        return self._load_size(self._load_test, batch_size=batch_size, batches=batches,
                               with_labels=with_labels, *args, **kwargs)

    def load_train(self, with_labels=True, batch_size=None, batches=None, *args, **kwargs):
        return self._load_size(self._load_train, batch_size=batch_size, batches=batches,
                               with_labels=with_labels, *args, **kwargs)

    def load_val(self, with_labels=True, batch_size=None, batches=None, *args, **kwargs):
        return self._load_size(self._load_val, batch_size=batch_size, batches=batches,
                               with_labels=with_labels, *args, **kwargs)

    def load_all_test(self, with_labels=False, *args, **kwargs):
        data = self._load_test(with_labels=with_labels, *args, **kwargs)
        for i in range(len(data)):
            yield next(data)

    def load_all_train(self, with_labels=True, *args, **kwargs):
        data = self._load_train(with_labels=with_labels, *args, **kwargs)
        for i in range(len(data)):
            yield next(data)

    def load_all_val(self, with_labels=True, *args, **kwargs):
        data = self._load_val(with_labels=with_labels, *args, **kwargs)
        for i in range(len(data)):
            yield next(data)

    def get_labels(self):
        labels = os.listdir(self.get_path(self._TRAIN_NAME))
        return sorted(labels)

    def transfer(self, path, with_labels=True, split=None):
        path = os.path.join(path, self.get_name())
        if with_labels:
            shutil.copytree(self.get_path(), path)
        else:
            cut = self.get_path().split(os.sep)
            if split is None:
                os.makedirs(path)
            else:
                for i in range(split):
                    os.makedirs(path + "_" + str(i))
            count = 0
            for root, dirs, files in os.walk(self.get_path()):
                suf = "_".join(root.split(os.sep)[len(cut):])
                for f in files:

                    src = os.path.join(root, f)
                    if split is None:
                        dst = os.path.join(path, "_".join([suf, f]))
                    else:
                        if count >= split:
                            count = 0
                        dst = os.path.join(path + "_" + str(count), "_".join([suf, f]))
                    # print(src, dst)
                    shutil.copy(src, dst)
                    count +=1

    def transfer_label(self, path, label):
        path = os.path.join(path, label)
        os.makedirs(path, exist_ok=True)
        for i in os.listdir(self.get_path()):
            src_dir = os.path.join(self.get_path(i), label)
            for f in os.listdir(src_dir):
                src = os.path.join(src_dir, f)
                dst = os.path.join(path, "_".join([i, f]))
                shutil.copy(src, dst)


class ImgDataSet(DataSet):

    _NAME = "ids"
    _CM = "categorical"

    def __init__(self, root=None, overwrite=False, shuffle=True, seed=123, batch_size=30, validation_split=0.1,
                 rescale=1/255, size=(224, 224), color="rgb", class_mode=None):
        super().__init__(root, overwrite, shuffle, seed, batch_size, validation_split)
        self.rescale = rescale
        self.size = size
        self.color = color
        self.class_mode = self._CM if class_mode is None else class_mode

    def _write_from_data(self, name, data, labels):
        classes = sorted(list(np.unique(labels)))
        # print(classes)
        for i in classes:
            os.makedirs(os.path.join(self.get_path(self._TEMP_NAME), name, str(i)))
        counter = {k: 0 for k in classes}
        for i in range(len(labels)):
            x = data[i]
            y = labels[i]
            counter[y] += 1
            # print(y)
            # print(os.path.join(self.get_path(self._TEMP_NAME), name, str(y), str(counter[y]) + ".png"))
            # cv2.imshow("img", x)
            # cv2.waitKey(0)
            tf.keras.preprocessing.image.save_img(os.path.join(self.get_path(self._TEMP_NAME),
                                                               name, str(y), str(counter[y]) + ".png"), x)

    def _create_from_data(self, test_data=None, trainings_data=None, validation_data=None):
        self._manage_temp(True)
        if test_data is not None:
            self._write_from_data(self._TEST_NAME, *test_data)
        if trainings_data is not None:
            self._write_from_data(self._TRAIN_NAME, *trainings_data)
        if validation_data is not None:
            self._write_from_data(self._VAL_NAME, *validation_data)
        self._create_from_dir(self.get_path(self._TEMP_NAME))
        self._manage_temp(False)

    def _load(self, gen, name, with_labels=True, subset=None, shuffle=None):
        if with_labels:
            class_mode = self.class_mode
        else:
            class_mode = None
        if shuffle is None:
            shuffle = self.shuffle
        return gen.flow_from_directory(directory=self.get_path(name), target_size=self.size,
                                       color_mode=self.color, class_mode=class_mode, batch_size=self.batch_size,
                                       shuffle=shuffle, seed=self.seed, subset=subset)

    def _load_test(self, with_labels=False, *args, **kwargs):
        gen = ImageDataGenerator(rescale=self.rescale, dtype=tf.float32)
        return self._load(gen, self._TEST_NAME, with_labels=with_labels)

    def _load_train(self, with_labels=True, *args, **kwargs):
        if self._val_exists():
            gen = ImageDataGenerator(rescale=self.rescale, dtype=tf.float32)
            return self._load(gen, self._TRAIN_NAME, with_labels=with_labels)
        else:
            gen = ImageDataGenerator(rescale=self.rescale, dtype=tf.float32, validation_split=self.validation_split)
            return self._load(gen, self._TRAIN_NAME, subset=self._TRAIN_NAME, with_labels=with_labels)

    def _load_val(self, with_labels=True, *args, **kwargs):
        if self._val_exists():
            gen = ImageDataGenerator(rescale=self.rescale, dtype=tf.float32)
            return self._load(gen, self._VAL_NAME, with_labels=with_labels)
        else:
            gen = ImageDataGenerator(rescale=self.rescale, dtype=tf.float32, validation_split=self.validation_split)
            return self._load(gen, self._TRAIN_NAME, subset=self._VAL_NAME, with_labels=with_labels)


class CIFAR10(ImgDataSet):

    _NAME = "cifar10"

    def _preprocess(self, data, labels):
        return data, labels.squeeze()

    def _create(self):
        train_data, test_data = tf.keras.datasets.cifar10.load_data()
        self._create_from_data(self._preprocess(*test_data), self._preprocess(*train_data))

class RCIFAR10(ImgDataSet):

    _NAME = "rcifar10"

    @staticmethod
    def _rename(label, color):
        return color + "_" + str(label)

    def _preprocess(self, data, labels):
        return self._preprocess_x(data), self._preprocess_y(labels)

    def _preprocess_x(self, data):
        return np.array([utils.change_hue_rgb_image(img, "red") for img in data])

    def _preprocess_y(self, labels):
        return labels.squeeze()

    def _create(self):
        train_data, test_data = tf.keras.datasets.cifar10.load_data()
        self._create_from_data(self._preprocess(*test_data), self._preprocess(*train_data))


class BCIFAR10(RCIFAR10):

    _NAME = "bcifar10"

    def _preprocess_x(self, data):
        return np.array([utils.change_hue_rgb_image(img, "blue") for img in data])


class GCIFAR10(RCIFAR10):

    _NAME = "gcifar10"

    def _preprocess_x(self, data):
        return np.array([utils.change_hue_rgb_image(img, "green") for img in data])


class CMNIST(ImgDataSet):

    _NAME = "cmnist"

    _COLORS = ["red", "green", "blue", "aqua", "pink", "yellow", "white"]

    @staticmethod
    def _rename(label, color):
        return color + "_" + str(label)

    def _preprocess(self, data, labels, colors=None):
        if colors is None:
            colors = self._COLORS
        return self._preprocess_x(data, colors), self._preprocess_y(labels, colors)

    def _preprocess_x(self, data, colors):
        return np.array([utils.color_bw_image(img, c) for c in colors for img in data])

    def _preprocess_y(self, labels, colors):
        return np.array([self._rename(i, c) for c in colors for i in labels])

    def _create(self):
        train_data, test_data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        self._create_from_data(self._preprocess(*test_data), self._preprocess(*train_data))


class CMNISTR1(CMNIST):

    _NAME = "rand_cmist1"


class CMNISTR2(CMNIST):
    _NAME = "rand_cmist2"


class CMNISTR3(CMNIST):
    _NAME = "rand_cmist3"


class CMNISTR4(CMNIST):
    _NAME = "rand_cmist4"



class CMNIST1(CMNIST):

    _NAME = "cmnist_red"
    _COLORS = ["red"]


class CMNISTSelect(CMNIST):

    _NAME = "cmnist_select"

    _COLORS = ["red", "green", "blue", "aqua", "pink", "yellow", "white"]

    def _rename(self, label, color):
        return self._COLORS.index(color)


class CMNIST2(CMNIST):

    _NAME = "cmnist_green"
    _COLORS = ["green"]


class CMNIST3(CMNIST):

    _NAME = "cmnist_blue"
    _COLORS = ["blue"]


class CMNIST4(CMNIST):

    _NAME = "cmnist_aqua"
    _COLORS = ["aqua"]


class CMNIST5(CMNIST):

    _NAME = "cmnist_pink"
    _COLORS = ["pink"]


class CMNIST6(CMNIST):

    _NAME = "cmnist_yellow"
    _COLORS = ["yellow"]


class CMNIST7(CMNIST):

    _NAME = "cmnist_white"
    _COLORS = ["white"]


class SMD(CMNIST):
    _NAME = "smd"

    def __init__(self, root=None, overwrite=False, shuffle=True, seed=123, batch_size=30, validation_split=0.1,
                 rescale=1/255, size=(224, 224), color="rgb", class_mode="categorical"):
        super(SMD, self).__init__(root=root, overwrite=overwrite, shuffle=shuffle, seed=seed, batch_size=batch_size,
                                  validation_split=validation_split, rescale=rescale, size=size, color=color,
                                  class_mode=class_mode)

    @staticmethod
    def _rename(label, color):
        if label in [2, 3, 5]:
            return 1
        elif label in [1, 4, 7]:
            return 2
        elif label in [6, 9, 0]:
            return 3
        else:
            return 4


class BinCMNIST(CMNIST):

    def __init__(self, root=None, overwrite=False, shuffle=True, seed=123, batch_size=30, validation_split=0.1,
                 rescale=1/255, size=(224, 224), color="rgb", class_mode="binary"):
        super().__init__(root, overwrite, shuffle, seed, batch_size, validation_split,
                         rescale, size, color, class_mode)


class BRD(BinCMNIST):

    _NAME = "brd"

    _COLORS = ["red", "blue"]

    @staticmethod
    def _rename(label, color):
        return int(color != "red")


class CRD(BinCMNIST):
    _NAME = "crd"

    @staticmethod
    def _rename(label, color):
        return int(color != "red")


class SCD(BinCMNIST):
    _NAME = "scd"

    @staticmethod
    def _rename(label, color):
        return int(label in [0, 6, 8, 9])


class BGRD(BinCMNIST):
    _NAME = "bgrd"

    _COLORS = ["red", "blue", "green"]

    @staticmethod
    def _rename(label, color):
        if color == "red":
            return 0
        elif color == "green":
            return int(label > 5)
        else:
            return 1


class CMNISTConcept(CMNIST):
    _NAME = "cmnist_concept"
    _N = 100
    _M = 100
    _SHAPE = (28, 28)

    def get_concept_name(self):
        return self._NAME.split("_")[-1]

    def _concept_foo(self, *args, **kwargs):
        pass

    def _create(self):
        train = np.array([self._concept_foo(self._SHAPE, color=c) for c in self._COLORS for _ in range(self._N)])
        test = np.array([self._concept_foo(self._SHAPE, color=c) for c in self._COLORS for _ in range(self._M)])
        train_labels = np.array([random.randint(0, 1) for _ in range(len(train))])
        test_labels = np.array([random.randint(0, 1) for _ in range(len(test))])
        self._create_from_data((train, train_labels), (test, test_labels))


class CMNISTCircle(CMNISTConcept):
    _NAME = "cmnist_circle"

    def _concept_foo(self, *args, **kwargs):
        return utils.create_circle_img(*args, **kwargs)


class CMNISTCrescent(CMNISTConcept):
    _NAME = "cmnist_crescent"

    def _concept_foo(self, *args, **kwargs):
        return utils.create_crescent_img(*args, **kwargs)


class CMNISTHLine(CMNISTConcept):
    _NAME = "cmnist_hline"

    def _concept_foo(self, *args, **kwargs):
        return utils.create_line_img(*args, degree=0, **kwargs)


class CMNISTVLine(CMNISTConcept):
    _NAME = "cmnist_vline"

    def _concept_foo(self, *args, **kwargs):
        return utils.create_line_img(*args, degree=90, **kwargs)


class RGBMNIST(CMNIST):

    _NAME = "rgb_mnist_shape"
    _COLORS = ["red", "green", "blue"]

    def _rename(self, label, color):
        if label in [1, 4, 7]:
            base = 0
        elif label in [2, 3, 5]:
            base = 1
        else:
            base = 2
        map_dict = {"red": 1.5, "green": 0.75, "blue": 0.25}
        val = int(max(0, round(base - random.uniform(0, map_dict[color]), 0)))
        # print(label, color, val)
        return val


class RGBMNISTStrict(CMNIST):

    _NAME = "rgb_mnist_shape_strict"
    _COLORS = ["red", "green", "blue"]
    _CM = "sparse"

    def _rename(self, label, color):
        # print(self.get_path())
        if label in [1, 4, 7]:
            base = 0
        elif label in [2, 3, 5]:
            base = 1
        else:
            base = 2
        map_dict = {"red": 2, "green": 1, "blue": 0}
        val = int(max(0, round(base - map_dict[color], 0)))
        # print(label, color, val)
        return val


class RGBMNISTControll(CMNIST):

    _NAME = "rgb_mnist_shape_control"
    _COLORS = ["red", "green", "blue"]
    _CM = "sparse"

    def _rename(self, label, color):
        # print(self.get_path())
        if label in [1, 4, 7]:
            base = 0
        elif label in [2, 3, 5]:
            base = 1
        else:
            base = 2
        return base

class RGBMNISTControllSparse(CMNIST):

    _NAME = "rgb_mnist_shape_control_sparse"
    _COLORS = ["red", "green", "blue"]
    _CM = "sparse"

    def _rename(self, label, color):
        # print(self.get_path())
        if label in [1, 4, 7]:
            base = 0
        elif label in [2, 3, 5]:
            base = 1
        else:
            base = 2
        return base


class Shapes(ImgDataSet):

    _NAME = "shapes"
    _SHAPE = (32, 32)
    _COLORS = ["red", "green", "blue"]
    _GEO_SHAPES = ["circle", "rectangle", "triangle"]
    _CM = "sparse"
    _N = 60000
    _M = 10000

    def _preprocess_y(self, shape, color):
        shape_map = {"circle": 2,
                     "rectangle": 1,
                     "triangle": 0}
        color_map = {"red": -1, "green": 0, "blue": 1}
        return min(2, max(0, shape_map[shape] + color_map[color]))

    def _create_shape(self, shape, color):
        foo_map = {"circle": utils.create_circle_img,
                   "rectangle": utils.create_rectangle_img,
                   "triangle": utils.create_triangle_img}
        return foo_map[shape](self._SHAPE, color=color, thickness=-1), self._preprocess_y(shape, color)

    def _create_set(self, n):
        n = int(n/3)
        labels = []
        data = []
        for c in self._COLORS:
            for s in self._GEO_SHAPES:
                for i in range(n):
                    x, y = self._create_shape(s, c)
                    data.append(x)
                    labels.append(y)
        return np.array(data), np.array(labels)

    def _create(self):
        train_x, train_y = self._create_set(self._N)
        test_x, test_y = self._create_set(self._M)
        self._create_from_data((train_x, train_y), (test_x, test_y))


class Shapes2(Shapes):

    _NAME = "shapes2"

    def _create_set(self, n):
        n = {"red": n/10, "blue": n/10, "green": n*8/10}
        labels = []
        data = []
        for c in self._COLORS:
            for s in self._GEO_SHAPES:
                for i in range(int(n[c])):
                    x, y = self._create_shape(s, c)
                    data.append(x)
                    labels.append(y)
        return np.array(data), np.array(labels)


class ShapesControll(Shapes):

    _NAME = "shapes_controll"

    def _preprocess_y(self, shape, color):
        shape_map = {"circle": 2,
                     "rectangle": 1,
                     "triangle": 0}
        color_map = {"red": 0, "green": 0, "blue": 0}
        return min(2, max(0, shape_map[shape] + color_map[color]))


class ShapesLabeled(Shapes):

    _NAME = "shape_labeled"
    shape_map = {"circle": 6,
                 "rectangle": 3,
                 "triangle": 0}
    color_map = {"red": 0, "green": 1, "blue": 2}

    def _initial_processing(self):
        self.translate_map = {int(v1 + v2): [k1, k2] for k1, v1 in self.shape_map.items() for k2, v2 in self.color_map.items()}

    def _preprocess_y(self, shape, color):
        return self.shape_map[shape] + self.color_map[color]

    def translate_label(self, label):
        return self.translate_map[int(label)]


class ShapeConcept(Shapes):
    _NAME = "shape_concept"
    _N = 5000
    _M = 5000


class ShapesRan1(ShapeConcept):

    _NAME = "shapes_ran1"


class ShapesRan2(ShapeConcept):

    _NAME = "shapes_ran2"


class ShapesRan3(ShapeConcept):

    _NAME = "shapes_ran3"


class Circle(ShapeConcept):

    _NAME = "circle"
    _SHAPE = (32, 32)
    _COLORS = ["red", "green", "blue"]
    _GEO_SHAPES = ["circle", "rectangle", "triangle"]


class Rectangle(ShapeConcept):

    _NAME = "rectangle"
    _SHAPE = (32, 32)
    _COLORS = ["red", "green", "blue"]
    _GEO_SHAPES = ["rectangle"]


class Triangle(ShapeConcept):

    _NAME = "triangle"
    _SHAPE = (32, 32)
    _COLORS = ["red", "green", "blue"]
    _GEO_SHAPES = ["triangle"]


class Red(ShapeConcept):

    _NAME = "red"
    _SHAPE = (32, 32)
    _COLORS = ["red"]
    _GEO_SHAPES = ["circle", "rectangle", "triangle"]


class Green(ShapeConcept):

    _NAME = "green"
    _SHAPE = (32, 32)
    _COLORS = ["green"]
    _GEO_SHAPES = ["circle", "rectangle", "triangle"]


class Blue(ShapeConcept):

    _NAME = "blue"
    _SHAPE = (32, 32)
    _COLORS = ["blue"]
    _GEO_SHAPES = ["circle", "rectangle", "triangle"]


class RedCircle(ShapeConcept):

    _NAME = "red_circle"
    _SHAPE = (32, 32)
    _COLORS = ["red"]
    _GEO_SHAPES = ["circle"]


class GreenCircle(ShapeConcept):

    _NAME = "green_circle"
    _SHAPE = (32, 32)
    _COLORS = ["green"]
    _GEO_SHAPES = ["circle"]


class BlueCircle(ShapeConcept):

    _NAME = "blue_circle"
    _SHAPE = (32, 32)
    _COLORS = ["blue"]
    _GEO_SHAPES = ["circle"]


class RedRectangle(ShapeConcept):
    _NAME = "red_rectangle"
    _SHAPE = (32, 32)
    _COLORS = ["red"]
    _GEO_SHAPES = ["rectangle"]


class GreenRectangle(ShapeConcept):
    _NAME = "green_rectangle"
    _SHAPE = (32, 32)
    _COLORS = ["green"]
    _GEO_SHAPES = ["rectangle"]


class BlueRectangle(ShapeConcept):
    _NAME = "blue_rectangle"
    _SHAPE = (32, 32)
    _COLORS = ["blue"]
    _GEO_SHAPES = ["rectangle"]


class RedTriangle(ShapeConcept):
    _NAME = "red_triangle"
    _SHAPE = (32, 32)
    _COLORS = ["red"]
    _GEO_SHAPES = ["triangle"]


class GreenTriangle(ShapeConcept):
    _NAME = "green_triangle"
    _SHAPE = (32, 32)
    _COLORS = ["green"]
    _GEO_SHAPES = ["triangle"]


class BlueTriangle(ShapeConcept):
    _NAME = "blue_triangle"
    _SHAPE = (32, 32)
    _COLORS = ["blue"]
    _GEO_SHAPES = ["triangle"]
