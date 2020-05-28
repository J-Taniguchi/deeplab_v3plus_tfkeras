import os
import sys
import yaml
import random

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)
use_devices = str(conf["use_devices"])
os.environ["CUDA_VISIBLE_DEVICES"] ="" # use_devices
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from tqdm import tqdm
from PIL import Image
import tensorflow as tf

from deeplab_v3plus_tfkeras.data_augment import augmentor
from deeplab_v3plus_tfkeras.input_data_processing import make_xy_path_list

tf.compat.v1.enable_eager_execution()

out_dir = os.path.join(conf["model_dir"], "augment_test")
image_size = conf["image_size"]

n_gpus = len(use_devices.split(','))

os.makedirs(out_dir, exist_ok=True)

train_x_paths, train_y_paths = make_xy_path_list(conf["train_x_paths"], conf["train_y_paths"])

n_out = 100
for i in tqdm(range(n_out - 1)):
    tar_idx = random.randint(0, len(train_x_paths) - 1)
    image = tf.io.read_file(train_x_paths[tar_idx])
    image = tf.image.decode_image(image, channels=3)

    mask = tf.io.read_file(train_y_paths[tar_idx])
    mask = tf.image.decode_image(mask, channels=3)

    image, _ = augmentor(image, mask)

    out_img = Image.fromarray(image.numpy())
    fpath = os.path.join(out_dir, "{:0>6}.png".format(i))
    out_img.save(fpath)