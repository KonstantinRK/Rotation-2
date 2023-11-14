"""Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
from six.moves import zip
import numpy as np
import six
import tensorflow as tf
from google.protobuf import text_format


class ModelWrapper(six.with_metaclass(ABCMeta, object)):
    """Simple wrapper of the for models with session object for TCAV.

      Supports easy inference with no need to deal with the feed_dicts.
    """

    def __init__(self, model=None, model_name=None, labels=None, node_dict=None):
        """Initialize the wrapper.

        Optionally create a session, load
        the model from model_path to this session, and map the
        input/output and bottleneck tensors.

        Args:
          model: one of the following: 1) Directory path to checkpoint 2)
            Directory path to SavedModel 3) File path to frozen graph.pb 4) File
            path to frozen graph.pbtxt
          node_dict: mapping from a short name to full input/output and bottleneck
            tensor names. Users should pass 'input' and 'prediction'
            as keys and the corresponding input and prediction tensor
            names as values in node_dict. Users can additionally pass bottleneck
            tensor names for which gradient Ops will be added later.
        """
        self.model_name = model_name
        self.model = model
        self.labels = labels
        self.bottleneck_map = {}

    def get_gradient(self, acts, y, bottleneck_name, example):
        """Return the gradient of the loss with respect to the bottleneck_name.

        Args:
          acts: activation of the bottleneck
          y: index of the logit layer
          bottleneck_name: name of the bottleneck to get gradient wrt.
          example: input example. Unused by default. Necessary for getting gradients
            from certain models, such as BERT.

        Returns:
          the gradient array.
        """
        x_tensor = tf.convert_to_tensor(acts, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            loss = self.loss_my(x_tensor, bottleneck_name, y)
        grad = t.gradient(loss, x_tensor).numpy()
        return grad

    @tf.function
    def loss_my(self, act, bottleneck_name, y):
        for layer in self.model.layers[self._tbn(bottleneck_name)+1:]:
            act = layer(act)
        return self.model.loss(y, act)

    def get_predictions(self, examples):
        """Get prediction of the examples.

        Args:
          imgs: array of examples to get predictions

        Returns:
          array of predictions
        """
        return self.model.predict(examples)

    def adjust_prediction(self, pred_t):
        """Adjust the prediction tensor to be the expected shape.

        Defaults to a no-op, but necessary to override for GoogleNet
        Returns:
          pred_t: pred_tensor.
        """
        return pred_t

    def reshape_activations(self, layer_acts):
        """Reshapes layer activations as needed to feed through the model network.

        Override this for models that require reshaping of the activations for use
        in TCAV.

        Args:
          layer_acts: Activations as returned by run_examples.

        Returns:
          Activations in model-dependent form; the default is a squeezed array (i.e.
          at most one dimensions of size 1).
        """
        return np.asarray(layer_acts).squeeze()

    def label_to_id(self, label):
        """Convert label (string) to index in the logit layer (id).

        Override this method if label to id mapping is known. Otherwise,
        default id 0 is used.
        """
        return self.labels.index(label)

    def id_to_label(self, idx):
        """Convert index in the logit layer (id) to label (string).

        Override this method if id to label mapping is known.
        """
        return self.labels[idx]

    def run_examples(self, examples, bottleneck_name):
        """Get activations at a bottleneck for provided examples.

        Args:
          examples: example data to feed into network.
          bottleneck_name: string, should be key of self.bottlenecks_tensors

        Returns:
          Activations in the given layer.
        """
        model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.layers[self._tbn(bottleneck_name)].output)
        return model.predict(examples)

    def _tbn(self, bottleneck_name):
        if isinstance(bottleneck_name, str):
            return self.bottleneck_map[bottleneck_name]
        else:
            return bottleneck_name



class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object."""

    def __init__(self, model, labels, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape)
        self.model = model
        self.labels = labels


class MobilenetV2Wrapper_public(PublicImageModelWrapper):

    def __init__(self, model, labels):
        self.image_value_range = (-1, 1)
        image_shape_v2 = [224, 224, 3]
        endpoints_v2 = dict(
            input='input:0',
            prediction='MobilenetV2/Predictions/Reshape:0',
        )
        super(MobilenetV2Wrapper_public, self).__init__(
            model,
            labels,
            image_shape_v2)

        self.model_name = 'MobilenetV2_public'


class GoogleNetWrapper_public(PublicImageModelWrapper):

    def __init__(self, model, labels):
        self.image_value_range = (-117, 255 - 117)
        image_shape_v2 = [224, 224, 3]
        endpoints_v2 = dict(
            input='input:0',
            prediction='MobilenetV2/Predictions/Reshape:0',
        )
        model = tf.keras.applications.InceptionV3()
        super(GoogleNetWrapper_public, self).__init__(
            model,
            labels,
            image_shape_v2)
        self.model_name = 'GoogleNet_public'


class GenericWrapper(PublicImageModelWrapper):

    def __init__(self, model, labels, img_value_range=(0, 1), shape=(32, 32, 3), model_name="generic_name"):
        self.image_value_range = img_value_range
        image_shape_v2 = shape
        super(GenericWrapper, self).__init__(
            model,
            labels,
            image_shape_v2)
        self.bottleneck_map = {l.name: i for i, l in enumerate(self.model.layers)}
        self.model_name = model_name