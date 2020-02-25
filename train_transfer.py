import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import glob
matplotlib.use('Agg')

model_dir = "./test3"
out_dir = "./test4"
traindata_dir = '../../data/train_data'
validdata_dir = '../../data/train_data_cut'
batch_size=8
n_epochs=1500
output_activation="sigmoid"

deeplabv3plus_srcdir="./src"
sys.path.append(deeplabv3plus_srcdir)

gpu_options = tf.compat.v1.GPUOptions(visible_device_list="3", allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options = gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


from model import deeplab_v3plus_transfer_os16
from data_gen import DataGenerator
from metrics import IoU
from label import Label
from loss import make_overwrap_crossentropy
from tensorflow.keras.utils import get_custom_objects

os.makedirs(out_dir, exist_ok=True)

train_x_paths = glob.glob(os.path.join(traindata_dir,'*.png'))
train_x_paths.sort()
image_names = [os.path.basename(train_x_paths[i]).split('.')[0] for i in range(len(train_x_paths))]
train_y_paths=[]
for i, image_name in enumerate(image_names):
    p = os.path.join(traindata_dir, image_name+'.json')
    if os.path.exists(p):
        train_y_paths.append(p)
    else:
        train_y_paths.append(None)

valid_x_paths = glob.glob(os.path.join(validdata_dir,'*.png'))
valid_x_paths.sort()
image_names = [os.path.basename(valid_x_paths[i]).split('.')[0] for i in range(len(valid_x_paths))]
valid_y_paths=[]
for i, image_name in enumerate(image_names):
    p = os.path.join(validdata_dir, image_name+'.json')
    if os.path.exists(p):
        valid_y_paths.append(p)
    else:
        valid_y_paths.append(None)


label_file_path = os.path.join(traindata_dir, 'label.csv')
label = Label(label_file_path)
get_custom_objects()["IoU"] = IoU
get_custom_objects()["overwrap_crossentropy"] = make_overwrap_crossentropy(label.n_labels)
image_size = (512,512)
model = keras.models.load_model(os.path.join(model_dir,'best_model.h5'))
model.summary()
preprocess = keras.applications.xception.preprocess_input



train_data_gen = DataGenerator(train_x_paths,
                               train_y_paths,
                               image_size,
                               label,
                               batch_size,
                               preprocess,
                               augmentation=True,
                               shuffle=True,
                               resize_or_crop="crop",
                               data_type="polygon")

valid_data_gen = DataGenerator(valid_x_paths,
                               valid_y_paths,
                               image_size,
                               label,
                               batch_size,
                               preprocess,
                               augmentation=False,
                               shuffle=False,
                               resize_or_crop="crop",
                               data_type="polygon")

if output_activation == "softmax":
    loss_function = tf.keras.losses.categorical_crossentropy
elif output_activation == "sigmoid":
    loss_function = make_overwrap_crossentropy(label.n_labels)
    #loss_function = tf.keras.losses.MSE
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss=loss_function, metrics=[IoU])

filepath = os.path.join(out_dir,'best_model.h5')
cp_cb = keras.callbacks.ModelCheckpoint(filepath,
                                        #monitor='IoU',
                                        monitor='val_IoU',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='max')


hist = model.fit_generator(train_data_gen,
                           epochs=n_epochs,
                           steps_per_epoch=len(train_data_gen),
                           validation_data=valid_data_gen,
                           validation_steps=len(valid_data_gen),
                           #shuffle = False,
                           workers=8,
                           use_multiprocessing=True,
                           callbacks=[cp_cb])

                           
                           
                           
                           
                           
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(hist.history["IoU"], label="IoU")
plt.plot(hist.history["val_IoU"], label="val_IoU")
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir,'losscurve.png'))


model.save(os.path.join(out_dir,'final_epoch.h5'))
for key in sorted(hist.history.keys()):
    np.savetxt(os.path.join(out_dir,key+'.txt'),np.array(hist.history[key]))
