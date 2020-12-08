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
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from deeplab_v3plus_tfkeras.metrics import make_IoU, make_categorical_IoU, make_F1score, make_categorical_F1score
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.input_data_processing import make_dataset
import deeplab_v3plus_tfkeras.loss as my_loss_func
tf.compat.v1.enable_eager_execution()
matplotlib.use('Agg')

out_dir = conf["model_dir"]
model_dir = conf["model_dir"]

label_file_path = conf["label_file_path"]

n_extra_channels = conf.get("n_extra_channels", 0)

train_x_dirs = conf["train_x_dirs"]
train_extra_x_dirs = conf.get("train_extra_x_dirs", None)
train_y_dirs = conf["train_y_dirs"]

valid_x_dirs = conf["valid_x_dirs"]
valid_extra_x_dirs = conf.get("valid_extra_x_dirs", None)
valid_y_dirs = conf["valid_y_dirs"]

batch_size = conf["batch_size"]
n_epochs = conf["n_epochs"]
output_activation = conf["output_activation"]
image_size = conf["image_size"]
loss = conf["loss"]
optimizer = conf["optimizer"]
metrics = conf["metrics"]
check_categorical_metrics = conf.get("check_categorical_metrics", "True")
class_weight = conf.get("class_weight", None)
use_tensorboard = conf.get("use_tensorboard", False)
use_batch_renorm = conf.get("use_batch_renorm", False)

label = Label(label_file_path)
if class_weight is not None:
    label.add_class_weight(class_weight)

hists_old = pd.read_csv(os.path.join(model_dir, "training_log.csv"))
initial_epoch = len(hists_old)

n_gpus = len(use_devices.split(','))
batch_size = batch_size * n_gpus

preprocess = keras.applications.xception.preprocess_input

# make train dataset
train_dataset, train_path_list = make_dataset(
    train_x_dirs,
    image_size,
    label,
    preprocess,
    batch_size,
    y_dirs=train_y_dirs,
    extra_x_dirs=train_extra_x_dirs,
    n_extra_channels=n_extra_channels,
    data_augment=True,
    shuffle=True
)

# make valid dataset
valid_dataset, valid_path_list = make_dataset(
    valid_x_dirs,
    image_size,
    label,
    preprocess,
    batch_size,
    y_dirs=valid_y_dirs,
    extra_x_dirs=valid_extra_x_dirs,
    n_extra_channels=n_extra_channels,
    data_augment=False,
    shuffle=False
)

# define loss function
if output_activation == "softmax":
    if loss == "CE":
        loss_function = tf.keras.losses.categorical_crossentropy
    elif loss == "FL":
        fl_alpha_list = [0.25] * label.n_labels
        fl_gamma_list = [2.0] * label.n_labels
        loss_function = \
            my_loss_func.make_focal_loss(label.n_labels,
                                         fl_alpha_list,
                                         fl_gamma_list)
    elif loss == "GDL":
        loss_function = my_loss_func.generalized_dice_loss
    else:
        raise Exception(loss + " is not supported.")

elif output_activation == "sigmoid":
    if loss == "CE":
        loss_function = keras.losses.binary_crossentropy
    elif loss == "FL":
        fl_alpha_list = [0.25] * label.n_labels
        fl_gamma_list = [2.0] * label.n_labels
        loss_function = \
            my_loss_func.make_focal_loss(label.n_labels,
                                         fl_alpha_list,
                                         fl_gamma_list)

    elif loss == "GDL":
        loss_function = my_loss_func.generalized_dice_loss
    else:
        raise Exception(loss + " is not supported.")

# define optimizer
if optimizer == "Adam":
    opt = tf.keras.optimizers.Adam()
elif optimizer == "Nadam":
    opt = tf.keras.optimizers.Nadam()
elif optimizer == "SGD":
    opt = tf.keras.optimizers.SGD()
else:
    raise Exception(
        "optimizer " + optimizer + " is not supported")

# define metrics
if metrics == "IoU":
    IoU = make_IoU(threshold=0.5)
    metrics_list = [IoU]
    if check_categorical_metrics:
        IoUs = make_categorical_IoU(label, threshold=0.5)
        metrics_list.extend(IoUs)
elif metrics == "F1score":
    F1 = make_F1score(threshold=0.5)
    metrics_list = [F1]
    if check_categorical_metrics:
        F1s = make_categorical_F1score(label, threshold=0.5)
        metrics_list.extend(F1s)
else:
    raise Exception(
        "metrics " + metrics + " is not supported")

# make model
model_file = os.path.join(model_dir, 'final_epoch.h5')
if n_gpus >= 2:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = keras.models.load_model(model_file, compile=False)
        model.compile(optimizer=opt,
                      loss=loss_function,
                      metrics=metrics_list,
                      run_eagerly=True)

else:
    model = keras.models.load_model(model_file, compile=False)
    model.compile(optimizer=opt,
                  loss=loss_function,
                  metrics=metrics_list,
                  run_eagerly=True)
model.summary()

filepath = os.path.join(model_dir, 'best_model.h5')
cp_cb = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_' + metrics,
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max')
cp_cb.best = hists_old["val_" + metrics].max()
cbs = [cp_cb]

if use_tensorboard:
    log_dir = os.path.join(out_dir, "logs")
    TB_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=False)
    cbs.append(TB_cb)

# training
n_train_data = len(train_path_list["x"])
n_valid_data = len(valid_path_list["x"])

n_train_batch = int(np.ceil(n_train_data / batch_size))
n_valid_batch = int(np.ceil(n_valid_data / batch_size))
print("train batch:{}".format(n_train_batch))
print("valid batch:{}".format(n_valid_batch))
hist = model.fit(
    train_dataset,
    epochs=n_epochs + initial_epoch,
    validation_data=valid_dataset,
    initial_epoch=initial_epoch,
    callbacks=cbs)

# write log
hists = hist.history
hists_new = pd.DataFrame(hists)

hists_df = pd.concat([hists_old, hists_new])
hists_df.reset_index(inplace=True, drop=True)

hists_df.to_csv(os.path.join(model_dir, "training_log.csv"), index=False)

if check_categorical_metrics:
    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.plot(hists_df["loss"], label="loss")
    plt.plot(hists_df["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(b=True)

    plt.subplot(1, 3, 2)
    for i, key in enumerate(hists_df):
        if 1 <= i <= 1 + label.n_labels:
            plt.plot(hists_df[key], label=key)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(b=True)

    plt.subplot(1, 3, 3)
    for i, key in enumerate(hists_df):
        if 3 + label.n_labels <= i:
            plt.plot(hists_df[key], label=key)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(b=True)
else:
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(hists_df["loss"], label="loss")
    plt.plot(hists_df["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(b=True)

    plt.subplot(1, 2, 2)
    plt.plot(hists_df[metrics], label=metrics)
    plt.plot(hists_df["val_" + metrics], label="val_" + metrics)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(b=True)

plt.savefig(os.path.join(out_dir, 'losscurve.png'))

model.save(os.path.join(out_dir, 'final_epoch.h5'))
