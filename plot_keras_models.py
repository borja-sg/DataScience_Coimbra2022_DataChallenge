import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

from keras.utils.vis_utils import plot_model
from keras.models import load_model


def _get_available_gpus():  
    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

#Input files
if tf.__version__ != '1.5.0':
    #Use this if we are using our own computer
    tfback._get_available_gpus = _get_available_gpus


PMTNUMBER = 1
traceLength = 5000

model_name = "CNN1_50epo"


model = load_model("./models/CNN1_50epo.h5")
#model2 = build_model_4ch_int
name = './imgs/models/CNN1_50epo_structure.pdf'
plot_model(model, to_file=name, show_shapes=True, show_layer_names=False, rankdir="TB")