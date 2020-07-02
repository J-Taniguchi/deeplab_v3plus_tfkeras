import os
import sys
import yaml

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)
use_devices = str(conf["use_devices"])
os.environ["CUDA_VISIBLE_DEVICES"] = use_devices
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import tensorflow.keras as keras
# from tensorflow.keras.utils import get_custom_objects
# import numpy as np
# from tqdm import tqdm

from deeplab_v3plus_tfkeras.data_utils import make_xy_from_data_paths
from deeplab_v3plus_tfkeras.data_utils import save_inference_results
from deeplab_v3plus_tfkeras.input_data_processing import make_data_path_list
from deeplab_v3plus_tfkeras.label import Label
# from deeplab_v3plus_tfkeras.metrics import make_IoU
# import deeplab_v3plus_tfkeras.loss as my_loss_func
# import deeplab_v3plus_tfkeras.data_gen as my_generator

model_dir = conf["model_dir"]

label_file_path = conf["label_file_path"]

batch_size = conf["batch_size"]
image_size = conf["image_size"]

n_extra_channels = conf.get("n_extra_channels", 0)

train_x_dirs = conf["train_x_dirs"]
train_extra_x_dirs = conf.get("train_extra_x_dirs", None)
train_y_dirs = conf["train_y_dirs"]

valid_x_dirs = conf["valid_x_dirs"]
valid_extra_x_dirs = conf.get("valid_extra_x_dirs", None)
valid_y_dirs = conf["valid_y_dirs"]

test_x_dirs = conf["test_x_dirs"]
test_extra_x_dirs = conf.get("test_extra_x_dirs", None)

model_for_inference = conf.get("model_for_inference", "best_model")
which_to_inference = conf["which_to_inference"]
label = Label(label_file_path)
n_gpus = len(use_devices.split(','))

batch_size = batch_size * n_gpus

model_file = os.path.join(model_dir, model_for_inference + '.h5')

if n_gpus >= 2:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = keras.models.load_model(model_file, compile=False)

else:
    model = keras.models.load_model(model_file, compile=False)

model.summary()

preprocess = keras.applications.xception.preprocess_input
last_activation = model.layers[-1].name

if "train" in which_to_inference:
    if n_extra_channels == 0:
        x_paths, y_paths, basenames = make_data_path_list(train_x_dirs, train_y_dirs, return_basenames=True)
        x, y = make_xy_from_data_paths(
            x_paths,
            y_paths,
            image_size,
            label)
        pred = model.predict(preprocess(x), batch_size=batch_size)
        fpath = os.path.join(model_dir, "train_inference.h5")
        save_inference_results(
            fpath,
            x,
            pred,
            last_activation,
            y=y,
            basenames=basenames)
    else:
        x_paths, y_paths, extra_x_paths, basenames =\
            make_data_path_list(
                train_x_dirs,
                train_y_dirs,
                extra_x_paths=train_extra_x_dirs,
                return_basenames=True)
        x, extra_x, y = make_xy_from_data_paths(
            x_paths,
            y_paths,
            image_size,
            label,
            extra_x_paths=extra_x_paths)
        pred = model.predict([preprocess(x), extra_x], batch_size=batch_size)
        fpath = os.path.join(model_dir, "train_inference.h5")
        save_inference_results(
            fpath,
            x,
            pred,
            last_activation,
            y=y,
            extra_x=extra_x,
            basenames=basenames)

if "valid" in which_to_inference:
    if n_extra_channels == 0:
        x_paths, y_paths, basenames = make_data_path_list(valid_x_dirs, valid_y_dirs, return_basenames=True)
        x, y = make_xy_from_data_paths(
            x_paths,
            y_paths,
            image_size,
            label)
        pred = model.predict(preprocess(x), batch_size=batch_size)
        fpath = os.path.join(model_dir, "valid_inference.h5")
        save_inference_results(
            fpath,
            x,
            pred,
            last_activation,
            y=y,
            basenames=basenames)
    else:
        x_paths, y_paths, extra_x_paths, basenames =\
            make_data_path_list(
                valid_x_dirs,
                valid_y_dirs,
                extra_x_paths=valid_extra_x_dirs,
                return_basenames=True)
        x, extra_x, y = make_xy_from_data_paths(
            x_paths,
            y_paths,
            image_size,
            label,
            extra_x_paths=extra_x_paths)
        pred = model.predict([preprocess(x), extra_x], batch_size=batch_size)
        fpath = os.path.join(model_dir, "valid_inference.h5")
        save_inference_results(
            fpath,
            x,
            pred,
            last_activation,
            y=y,
            extra_x=extra_x,
            basenames=basenames)

if "test" in which_to_inference:
    if n_extra_channels == 0:
        for i, test_x_dir in enumerate(test_x_dirs):
            test_name = test_x_dir.split(os.sep)[-1]
            x_paths, basenames = make_data_path_list([test_x_dir], return_basenames=True)
            x = make_xy_from_data_paths(
                x_paths,
                None,
                image_size,
                label)
            pred = model.predict(preprocess(x), batch_size=batch_size)
            fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
            save_inference_results(
                fpath,
                x,
                pred,
                last_activation,
                basenames=basenames)
    else:
        for i, test_x_dir in enumerate(test_x_dirs):
            test_name = test_x_dir.split(os.sep)[-1]
            x_paths, extra_x_paths, basenames =\
                make_data_path_list(
                    [test_x_dir],
                    extra_x_paths=[test_extra_x_dirs[i]],
                    return_basenames=True)
            x, extra_x = make_xy_from_data_paths(
                x_paths,
                None,
                image_size,
                label,
                extra_x_paths=extra_x_paths)
            pred = model.predict([preprocess(x), extra_x], batch_size=batch_size)
            fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
            save_inference_results(
                fpath,
                x,
                pred,
                last_activation,
                extra_x=extra_x,
                basenames=basenames)
