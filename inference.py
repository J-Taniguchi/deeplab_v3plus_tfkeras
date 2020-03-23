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
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from tqdm import tqdm

from deeplab_v3plus_tfkeras.data_utils import make_xy_from_data_paths
from deeplab_v3plus_tfkeras.data_utils import inference_large_img
from deeplab_v3plus_tfkeras.data_utils import save_inference_results
from deeplab_v3plus_tfkeras.input_data_processing import check_data_paths
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_path_list
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_array
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.metrics import make_IoU
import deeplab_v3plus_tfkeras.loss as my_loss_func
import deeplab_v3plus_tfkeras.data_gen as my_generator

model_dir = conf["model_dir"]

label_file_path = conf["label_file_path"]
train_data_paths = conf["train_data_paths"]
valid_data_paths = conf["valid_data_paths"]
test_data_paths = conf["test_data_paths"]

batch_size = conf["batch_size"]
image_size = conf["image_size"]

print(test_data_paths)
if train_data_paths is not None:
    train_data_types = check_data_paths(train_data_paths)
if valid_data_paths is not None:
    valid_data_types = check_data_paths(valid_data_paths)
if test_data_paths is not None:
    test_data_types = check_data_paths(test_data_paths)

# loss = conf["loss"]
which_to_inference = conf["which_to_inference"]
label = Label(label_file_path)
n_gpus = len(use_devices.split(','))

batch_size = batch_size * n_gpus

model_file = os.path.join(model_dir,'best_model.h5')
if n_gpus >=2:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = keras.models.load_model(model_file, compile=False)

else:
    model = keras.models.load_model(model_file, compile=False)

model.summary()

preprocess = keras.applications.xception.preprocess_input
last_activation = model.layers[-1].name


if "train" in which_to_inference:
    if train_data_types[0] == "dir":
        x_paths, y_paths = make_xy_path_list(train_data_paths)
        x, y = make_xy_from_data_paths(x_paths,
                                       y_paths,
                                       image_size,
                                       label,
                                       "polygon",
                                       resize_or_crop="crop")
    else:
        x, y = make_xy_array(train_data_paths)
    pred = model.predict(preprocess(x), batch_size=batch_size)
    fpath = os.path.join(model_dir, "train_inference.h5")
    save_inference_results(fpath,
                           x=x,
                           pred=pred,
                           y=y,
                           last_activation=last_activation)


if "valid" in which_to_inference:
    if valid_data_types[0] == "dir":
        x_paths, y_paths = make_xy_path_list(valid_data_paths)
        x, y = make_xy_from_data_paths(x_paths,
                                       y_paths,
                                       image_size,
                                       label,
                                       "polygon",
                                       resize_or_crop="crop")
    else:
        x, y = make_xy_array(valid_data_paths)

    pred = model.predict(preprocess(x), batch_size=batch_size)
    fpath = os.path.join(model_dir, "valid_inference.h5")
    save_inference_results(fpath,
                           x=x,
                           pred=pred,
                           y=y,
                           last_activation=last_activation)


if "test" in which_to_inference:
    for i, test_data_path in enumerate(test_data_paths):
        if test_data_types[i] == "dir":
            test_name = test_data_path.split(os.sep)[-1]
            x_paths, y_paths = make_xy_path_list([test_data_path])
            x, y = make_xy_from_data_paths(x_paths,
                                           y_paths,
                                           image_size,
                                           label,
                                           "polygon",
                                           resize_or_crop="crop")
        else:
            basename = os.path.basename(test_data_path)
            test_name = os.path.splitext(basename)
            x, y = make_xy_array([test_data_path])

        mode = "max_confidence"
        print(mode)
        x = []
        y =[]

        for x_path in tqdm(x_paths):
        #for i in tqdm(np.arange(0, len(valid_x_paths), 1000)):
            x0, y0 = inference_large_img(
                x_path,
                model,
                preprocess,
                mode=mode,
                threshold=0.5,
                batch_size=batch_size)
            x.append(x0)
            y.append(y0)

        x = np.array(x)
        y = np.array(y)

        fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
        save_inference_results(fpath,
                               x=x,
                               pred=y,
                               last_activation=last_activation)
