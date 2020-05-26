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
import numpy as np
from tqdm import tqdm

from deeplab_v3plus_tfkeras.data_utils import make_xy_from_data_paths
from deeplab_v3plus_tfkeras.data_utils import inference_large_img
from deeplab_v3plus_tfkeras.data_utils import save_inference_results
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_path_list
from deeplab_v3plus_tfkeras.label import Label


model_dir = conf["model_dir"]

label_file_path = conf["label_file_path"]

batch_size = conf["batch_size"]
image_size = conf["image_size"]


# loss = conf["loss"]
which_to_inference = conf["which_to_inference"]
label = Label(label_file_path)
n_gpus = len(use_devices.split(','))

batch_size = batch_size * n_gpus

model_file = os.path.join(model_dir, 'best_model.h5')
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
    x_paths, y_paths = make_xy_path_list(conf["train_x_paths"], conf["train_y_paths"])
    x, y = make_xy_from_data_paths(x_paths,
                                   y_paths,
                                   image_size,
                                   label,
                                   "image",
                                   resize_or_crop="crop")
    pred = model.predict(preprocess(x), batch_size=batch_size)
    fpath = os.path.join(model_dir, "train_inference.h5")
    save_inference_results(fpath,
                           x=x,
                           y=y,
                           pred=pred,
                           last_activation=last_activation)

if "valid" in which_to_inference:
    x_paths, y_paths = make_xy_path_list(conf["valid_x_paths"], conf["valid_y_paths"])
    x, y = make_xy_from_data_paths(x_paths,
                                   y_paths,
                                   image_size,
                                   label,
                                   "image",
                                   resize_or_crop="crop")
    pred = model.predict(preprocess(x), batch_size=batch_size)
    fpath = os.path.join(model_dir, "valid_inference.h5")
    save_inference_results(fpath,
                           x=x,
                           y=y,
                           pred=pred,
                           last_activation=last_activation)

if "test" in which_to_inference:
    for i, test_data_path in enumerate(conf["test_x_paths"]):
        test_name = test_data_path.split(os.sep)[-1]
        x_paths, _ = make_xy_path_list(conf["test_x_paths"], None)
        """
        x = make_xy_from_data_paths(x_paths,
                                    None,
                                    image_size,
                                    label,
                                    "image",
                                    resize_or_crop="crop")
        """

        mode = "max_confidence"
        x = []
        pred = []
        for x_path in tqdm(x_paths):
            x0, pred0 = inference_large_img(
                x_path,
                model,
                preprocess,
                mode=mode,
                threshold=0.5,
                batch_size=batch_size)
            x.append(x0)
            pred.append(pred0)

        x = np.array(x)
        pred = np.array(pred)

        fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
        save_inference_results(fpath,
                               x=x,
                               pred=pred,
                               last_activation=last_activation)
