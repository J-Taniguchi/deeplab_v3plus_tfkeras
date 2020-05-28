import os
import sys
import yaml
import shutil

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)
use_devices = str(conf["use_devices"])
os.environ["CUDA_VISIBLE_DEVICES"] = use_devices
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from deeplab_v3plus_tfkeras.model import deeplab_v3plus_transfer_os16
from deeplab_v3plus_tfkeras.metrics import make_IoU
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_path_list
import deeplab_v3plus_tfkeras.data_gen as my_generator
import deeplab_v3plus_tfkeras.loss as my_loss_func
tf.compat.v1.enable_eager_execution()
matplotlib.use('Agg')

out_dir = conf["model_dir"]
label_file_path = conf["label_file_path"]

train_x_dirs = conf["train_x_dirs"]
train_y_dirs = conf["train_y_dirs"]

valid_x_dirs = conf["valid_x_dirs"]
valid_y_dirs = conf["valid_y_dirs"]

batch_size = conf["batch_size"]
n_epochs = conf["n_epochs"]
output_activation = conf["output_activation"]
image_size = conf["image_size"]
loss = conf["loss"]
optimizer = conf["optimizer"]
class_weight = conf.get("class_weight", None)
use_tensorboard = conf["use_tensorboard"]

label = Label(label_file_path)
if class_weight is not None:
    label.add_class_weight(class_weight)

n_gpus = len(use_devices.split(','))

batch_size = batch_size * n_gpus

os.makedirs(out_dir, exist_ok=True)

preprocess = keras.applications.xception.preprocess_input

# make train dataset
train_x_paths, train_y_paths = make_xy_path_list(train_x_dirs, train_y_dirs)
n_train_data = len(train_x_paths)
train_dataset, train_map_f = my_generator.make_path_generator(
    train_x_paths,
    train_y_paths,
    image_size,
    label,
    preprocess,
    augmentation=True,
    # augmentation=False,
    resize_or_crop="crop",
    data_type="image")

train_dataset = train_dataset.shuffle(n_train_data)
train_dataset = train_dataset.map(train_map_f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# make valid dataset
valid_x_paths, valid_y_paths = make_xy_path_list(valid_x_dirs, valid_y_dirs)
n_valid_data = len(valid_x_paths)
valid_dataset, valid_map_f = my_generator.make_path_generator(
    valid_x_paths,
    valid_y_paths,
    image_size,
    label,
    preprocess,
    augmentation=False,
    resize_or_crop="crop",
    data_type="image")

valid_dataset = valid_dataset.map(valid_map_f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
IoU = make_IoU(threshold=0.5)

# make model
layer_name_to_decoder = "block3_sepconv2_bn"
encoder_end_layer_name = "block13_sepconv2_bn"

if n_gpus >= 2:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        encoder = keras.applications.Xception(
            input_shape=(*image_size, 3),
            weights="imagenet",
            include_top=False)

        model = deeplab_v3plus_transfer_os16(
            label.n_labels,
            encoder,
            layer_name_to_decoder,
            encoder_end_layer_name,
            freeze_encoder=False,
            output_activation=output_activation,
            batch_renorm=False)

        model.compile(optimizer=opt,
                      loss=loss_function,
                      metrics=[IoU],
                      run_eagerly=True,
                      )
else:
    encoder = keras.applications.Xception(
        input_shape=(*image_size, 3),
        weights="imagenet",
        include_top=False)

    model = deeplab_v3plus_transfer_os16(
        label.n_labels,
        encoder,
        layer_name_to_decoder,
        encoder_end_layer_name,
        freeze_encoder=False,
        output_activation=output_activation,
        batch_renorm=False,
    )

    model.compile(optimizer=opt,
                  loss=loss_function,
                  metrics=[IoU],
                  run_eagerly=True,
                  )
model.summary()

filepath = os.path.join(out_dir, 'best_model.h5')
cp_cb = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_IoU',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max')
cbs = [cp_cb]

if use_tensorboard:
    log_dir = os.path.join(out_dir, "logs")
    TB_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=False)
    cbs.append(TB_cb)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

# training
n_train_batch = int(np.ceil(n_train_data / batch_size))
n_valid_batch = int(np.ceil(n_valid_data / batch_size))
print("train batch:{}".format(n_train_batch))
print("valid batch:{}".format(n_valid_batch))
hist = model.fit(
    train_dataset,
    epochs=n_epochs,
    validation_data=valid_dataset,
    callbacks=cbs,
)

# write log
hists = [hist.history["loss"],
         hist.history["val_loss"],
         hist.history["IoU"],
         hist.history["val_IoU"]]
hists = np.array(hists).T
hists_df = pd.DataFrame(hists, columns=["loss", "val_loss", "IoU", "val_IoU"])
hists_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(hists_df["loss"], label="loss")
plt.plot(hists_df["val_loss"], label="val_loss")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(hists_df["IoU"], label="IoU")
plt.plot(hists_df["val_IoU"], label="val_IoU")
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, 'losscurve.png'))

model.save(os.path.join(out_dir, 'final_epoch.h5'))
