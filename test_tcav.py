import tcav2.activation_generator as act_gen
import tcav2.cav as cav
import tcav2.model as model
import tcav2.tcav as tcav
import tcav2.utils as utils
import tcav2.utils_plot as utils_plot # utils_plot requires matplotlib
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#
print('REMEMBER TO UPDATE YOUR_PATH (where images, models are)!')

# This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
model_to_run = 'GoogleNet'
user = 'beenkim'
# the name of the parent directory that results are stored (only if you want to cache)
project_name = 'tcav_class_test'
working_dir = "/tmp/" + user + '/' + project_name
# where activations are stored (only if your act_gen_wrapper does so)
activation_dir = working_dir + '/activations/'
# where CAVs are stored.
# You can say None if you don't wish to store any.
cav_dir = working_dir + '/cavs/'
# where the images live.

# network = tf.keras.applications.MobileNet()
# print(network.summary())
# print(network.layers[-2].name)
# TODO: replace 'YOUR_PATH' with path to downloaded models and images.
source_dir = "/home/krk/Dropbox/IST/year_1/rotations/rotation_2/downloads"
bottlenecks = [-5]  # @param

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.1]

target = 'zebra'
concepts = ["dotted", "striped", "zigzagged"]


# GRAPH_PATH is where the trained model is stored.
GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
# LABEL_PATH is where the labels are stored. Each line contains one class, and they are ordered with respect to their index in
# the logit layer. (yes, id_to_label function in the model wrapper reads from this file.)
# For example, imagenet_comp_graph_label_strings.txt looks like:
# dummy
# kit fox
# English setter
# Siberian husky ...

import test




# LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"
#
# labels = tf.io.gfile.GFile(LABEL_PATH).read().splitlines()
# mymodel = model.GoogleNetWrapper_public(None, labels)
#
# act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=15)
# tf.compat.v1.logging.set_verbosity(0)
# num_random_exp= 3
# ## only running num_random_exp = 10 to save some time. The paper number are reported for 500 random runs.
# mytcav = tcav.TCAV(target,
#                    concepts,
#                    bottlenecks,
#                    act_generator,
#                    alphas,
#                    cav_dir=cav_dir,
#                    num_random_exp=num_random_exp)#10)
# print('This may take a while... Go get coffee!')
# results = mytcav.run(run_parallel=False)
# utils_plot.plot_results(results, num_random_exp=num_random_exp)
#
#
# print('done!')