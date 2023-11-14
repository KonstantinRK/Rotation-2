import tensorflow as tf
import os
from utils import BaseClass


def is_monitored(foo):
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.monitors is not None:
            args, kwargs, self.monitors.monitor(foo.__name__, *args, **kwargs)
        return foo(*args, **kwargs)
    return wrapper


class NetworkWrapper(BaseClass):
    _NAME = "nnet"

    @staticmethod
    def _define_network(input_size, classes, activation):
        pass

    def __init__(self, inp_size=None, classes=None, root="", loss=None, *args, **kwargs):
        super(NetworkWrapper, self).__init__(root)
        self.model = None
        self.input_size = inp_size
        self.classes = classes
        self.trained_on = None
        self.status = -1
        self.loss = loss

    def is_ready(self):
        return self.status == 2

    def build_network(self, input_size=None, classes=None, activation=None):
        input_size = self.input_size if input_size is None else input_size
        classes = self.classes if classes is None else classes
        classes = classes if classes > 2 else 1
        if activation is None:
            activation = "sigmoid" if classes == 1 else "softmax"
        self.model = self._define_network(input_size, classes, activation)
        self.status = 0

    def compile(self):
        if self.loss is None:
            if self.classes < 3:
                loss = "binary_crossentropy"
            else:
                loss = "categorical_crossentropy"
        else:
            loss = self.loss
        self.compile_manual(optimizer="adam", loss=loss, metrics=["accuracy"])
        for i, l in enumerate(self.model.layers):
            print(i, l.name)
        print(self.model.summary())

    def compile_manual(self, optimizer, loss, metrics, *args, **kwargs):
        if self.status < 0:
            self.build_network()
        self.model.compile(optimizer, loss, metrics, *args, **kwargs)
        self.status = 1

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        if self.status < 1:
            self.compile()
        self.model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                       class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                       validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
        self.status = 2

    def predict_simple(self, x, batch_size=None, verbose=0, steps=None, callbacks=None,
                       max_queue_size=10, workers=1, use_multiprocessing=False):
        return self.model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None,
                max_queue_size=10, workers=1, use_multiprocessing=False):
        return self._predict_internal(x, batch_size, verbose, steps, callbacks, max_queue_size, workers,
                                     use_multiprocessing)[-1]

    def _predict_internal(self, x, batch_size=None, verbose=0, steps=None, callbacks=None,
                         max_queue_size=10, workers=1, use_multiprocessing=False):
        model = tf.keras.Model(inputs=self.model.inputs, outputs=[layer.output for layer in self.model.layers])
        return model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)

    def activations(self, x, layer=None, batch_size=None, verbose=0, steps=None, callbacks=None,
                    max_queue_size=10, workers=1, use_multiprocessing=False):
        data = self._predict_internal(x, batch_size, verbose, steps, callbacks, max_queue_size, workers,
                                      use_multiprocessing)
        pred = data[-1]
        if layer is not None:
            data = data[layer]
        return pred, data

    def save(self, dataset):
        dataset = self.resolve_dataset_name(dataset)
        if self.status >= 2:
            if not os.path.exists(self.get_path()):
                os.makedirs(self.get_path())
            self.model.save(self.get_path(dataset))
        else:
            raise ValueError("Untrained models can not be saved.")

    def load(self, dataset):
        dataset = self.resolve_dataset_name(dataset)
        if os.path.exists(self.get_path(dataset)):
            self.model = tf.keras.models.load_model(self.get_path(dataset))
            self.trained_on = dataset
            self.status = 2
            return True
        else:
            return False

    def get_layers(self):
        return self.model.layers

    @staticmethod
    def resolve_dataset_name(dataset):
        if isinstance(dataset, str):
            return dataset
        else:
            return dataset.get_name()


class LeNet5(NetworkWrapper):
    _NAME = "lenet"

    @staticmethod
    def _define_network(input_size, classes, activation):
        print(classes)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_size))
        model.add(tf.keras.layers.AveragePooling2D())
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=120, activation='relu'))
        model.add(tf.keras.layers.Dense(units=84, activation='relu'))
        model.add(tf.keras.layers.Dense(units=classes if classes > 2 else 1, activation=activation))
        return model

    def __init__(self, inp_size=(32, 32, 3), classes=2, root="", loss=None, *args, **kwargs):
        super(LeNet5, self).__init__(inp_size, classes, root, loss=loss, *args, **kwargs)


