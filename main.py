# from datasets import *
# from networks import LeNet5
# from monitors import BaseMonitor
# from pipelines import ImgPipeLine

import tensorflow as tf
from experiments import *
from manager import FCMManager
ROOT = "/home/krk/Documents/"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#
fcm = FCMManager(root=ROOT)
fcm.initialise_experiment(TcavFinCavMonitoringControl)
# # fcm.reset_experiment()
# # fcm.setup_experiment()
# # fcm.preprocess_experiment()
fcm.run_experiment(True)
# fcm.clean_experiment()
#
# import pickle
# with open("/home/krk/Documents/fcm/experiments/tcav_final_monitoring_shapes/saliency_tcav_monitoring/res/res.pkl", "rb") as f:
#     print(pickle.load(f))