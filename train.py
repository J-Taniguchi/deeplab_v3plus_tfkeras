import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import yaml
import glob
from deeplab_v3plus_tfkeras.model import deeplab_v3plus_transfer_os16
from deeplab_v3plus_tfkeras.metrics import make_IoU
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.input_data_processing import check_data_paths
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_path_list
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_array
import deeplab_v3plus_tfkeras.data_gen as my_generator
import deeplab_v3plus_tfkeras.loss as my_loss_func

matplotlib.use('Agg')

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)

out_dir = conf["model_dir"]
label_file_path = conf["label_file_path"]
train_data_paths = conf["train_data_paths"]
valid_data_paths = conf["valid_data_paths"]

#n_gpu = 4
batch_size = conf["batch_size"]
n_epochs = conf["n_epochs"]
output_activation = conf["output_activation"]
use_devise = str(conf["use_devise"])
image_size = conf["image_size"]
loss = conf["loss"]

label = Label(label_file_path)
train_data_types = check_data_paths(train_data_paths, mixed_type_is_error=True)
valid_data_types = check_data_paths(valid_data_paths, mixed_type_is_error=True)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#tf.compat.v1.enable_eager_execution()

gpu_options = tf.compat.v1.GPUOptions(
    visible_device_list=use_devise, allow_growth=True)
#gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0,1,2,3", allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


os.makedirs(out_dir, exist_ok=True)

preprocess = keras.applications.xception.preprocess_input
# make train data generator
if train_data_types[0] == "dir":
    train_x_paths, train_y_paths = make_xy_path_list(train_data_paths)
    train_data_gen = my_generator.path_DataGenerator(
        train_x_paths,
        train_y_paths,
        image_size,
        label,
        batch_size,
        preprocess,
        augmentation=True,
        shuffle=True,
        resize_or_crop="crop",
        data_type="polygon")
else:
    train_x, train_y = make_xy_array(train_data_paths)
    train_data_gen = my_generator.array_DataGenerator(
        train_x,
        train_y,
        batch_size,
        preprocess,
        augmentation=True,
        shuffle=True,
        )

# make valid data generator
if valid_data_types[0] == "dir":
    valid_x_paths, valid_y_paths = make_xy_path_list(valid_data_paths)
    valid_data_gen = my_generator.path_DataGenerator(
        valid_x_paths,
        valid_y_paths,
        image_size,
        label,
        batch_size,
        preprocess,
        augmentation=False,
        shuffle=False,
        resize_or_crop="crop",
        data_type="polygon")

else:
    valid_x, valid_y = make_xy_array(valid_data_paths)
    valid_data_gen = my_generator.array_DataGenerator(
        valid_x,
        valid_y,
        batch_size,
        preprocess,
        augmentation=False,
        shuffle=False,
        )

# make model
encoder = keras.applications.Xception(
    input_shape=(*image_size,3),
    weights="imagenet",
    include_top=False)

layer_name_to_decoder = "block3_sepconv2_bn"
encoder_end_layer_name = "block13_sepconv2_bn"

model = deeplab_v3plus_transfer_os16(
    label.n_labels,
    encoder,
    layer_name_to_decoder,
    encoder_end_layer_name,
    freeze_encoder=False,
    output_activation=output_activation,
    batch_renorm=True)
model.summary()

#multi_gpu_model = keras.utils.multi_gpu_model(model, gpus=n_gpu)

# define loss function
if output_activation == "softmax":
    loss_function = tf.keras.losses.categorical_crossentropy
elif output_activation == "sigmoid":
    if loss == "CE":
        loss_function = \
            my_loss_func.make_overwrap_crossentropy(label.n_labels)
    elif loss == "FL":
        alphas = [0.25, 0.25]
        gammas = [2.0, 2.0]
        loss_function = \
            my_loss_func.make_overwrap_focalloss(label.n_labels,
                                                 alphas, gammas)
    elif loss == "WCE":
        weights = [0.99, 0.99]
        loss_function = \
            my_loss_func.make_weighted_overwrap_crossentropy(label.n_labels,
                                                             weights)
    elif loss =="GDL":
        loss_function = my_loss_func.generalized_dice_loss
    else:
        raise Exception(loss+" is not supported.")

# define optimzer
#opt = tf.keras.optimizers.Adam()
opt = tf.keras.optimizers.Nadam()
#opt = tf.keras.optimizers.SGD()

IoU = make_IoU(threshold=0.5)
model.compile(optimizer=opt,
              loss=loss_function,
              metrics=[IoU],
              run_eagerly=True)

filepath = os.path.join(out_dir,'best_model.h5')
cp_cb = keras.callbacks.ModelCheckpoint(
    filepath,
    #monitor='IoU',
    monitor='val_IoU',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max')


hist = model.fit_generator(
    train_data_gen,
    epochs=n_epochs,
    steps_per_epoch=len(train_data_gen),
    validation_data=valid_data_gen,
    validation_steps=len(valid_data_gen),
    #shuffle = False,
    workers=8,
    use_multiprocessing=True,
    callbacks=[cp_cb])

hists = [hist.history["loss"],
         hist.history["val_loss"],
         hist.history["IoU"],
         hist.history["val_IoU"]]
hists = np.array(hists).T
hists_df = pd.DataFrame(hists, columns=["loss", "val_loss", "IoU", "val_IoU"])
hists_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.plot(hists_df["loss"], label="loss")
plt.plot(hists_df["val_loss"], label="val_loss")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(hists_df["IoU"], label="IoU")
plt.plot(hists_df["val_IoU"], label="val_IoU")
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir,'losscurve.png'))


model.save(os.path.join(out_dir,'final_epoch.h5'))
