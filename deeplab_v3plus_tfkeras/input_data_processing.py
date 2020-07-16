import os
# import sys

import tensorflow as tf
import h5py
import numpy as np
from glob import glob

from .data_gen import make_generator, make_generator_with_extra_x


def make_dataset(x_dirs,
                 image_size,
                 label,
                 preprocess,
                 batch_size,
                 y_dirs=None,
                 extra_x_dirs=None,
                 n_extra_channels=0,
                 data_augment=False,
                 shuffle=False):
    if n_extra_channels == 0:
        path_list = make_data_path_list(
            x_dirs,
            y_dirs=y_dirs)
        n_data = len(path_list["x"])
        dataset, map_f = make_generator(
            path_list["x"],
            image_size,
            label,
            seg_img_paths=path_list["y"],
            preprocess=preprocess,
            augmentation=data_augment,
        )
    else:
        path_list = make_data_path_list(
            x_dirs,
            y_dirs=y_dirs,
            extra_x_dirs=extra_x_dirs)
        n_data = len(path_list["x"])
        dataset, map_f = make_generator_with_extra_x(
            path_list["x"],
            path_list["extra_x"],
            image_size,
            label,
            seg_img_paths=path_list["y"],
            preprocess=preprocess,
            augmentation=data_augment)
    if shuffle:
        dataset = dataset.shuffle(n_data)
    dataset = dataset.map(map_f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, path_list


def check_data_paths(data_paths, mixed_type_is_error=False):
    is_path = False
    is_h5 = False
    out = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            is_path = True
            out.append("dir")
        elif os.path.isfile(data_path):
            is_h5 = True
            out.append("h5")

    if mixed_type_is_error and is_path and is_h5:
        print(data_paths)
        raise Exception("all data_paths must same type (dir or h5).")

    return out


def make_data_path_list(x_dirs,
                        y_dirs=None,
                        extra_x_dirs=None,
                        img_exts=["png", "jpg"]):
    x = []
    y = []
    basenames = []

    if extra_x_dirs is not None:
        extra_x = []
    for i, x_dir in enumerate(x_dirs):
        if y_dirs is not None:
            y_dir = y_dirs[i]
        if extra_x_dirs is not None:
            extra_x_dir = extra_x_dirs[i]

        x0 = []
        for ext in img_exts:
            x0.extend(glob(os.path.join(x_dir, '*.' + ext)))
        x0.sort()

        for fpath in x0:
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            if y_dirs is not None:
                p = os.path.join(y_dir, base_name + '.png')
                if os.path.exists(p):
                    y.append(p)
                else:
                    raise Exception("{} dose not exist.".format(p))
            if extra_x_dirs is not None:
                p = os.path.join(extra_x_dir, base_name + '.npy')
                if os.path.exists(p):
                    extra_x.append(p)
                else:
                    raise Exception("{} dose not exist.".format(p))
            basenames.append(base_name)
        x.extend(x0)

    path_list = dict()
    path_list["x"] = x

    if y_dirs is None:
        path_list["y"] = None
    else:
        path_list["y"] = y

    if extra_x_dirs is None:
        path_list["extra_x"] = None
    else:
        path_list["extra_x"] = extra_x

    path_list["basenames"] = basenames

    return path_list


def make_xy_array(data_paths):
    for i, data_path in enumerate(data_paths):
        if i == 0:
            with h5py.File(data_path, "r") as f:
                x = f["x"][:].astype(np.uint8)
                y = f["y"][:].astype(np.float32)
        else:
            with h5py.File(data_path, "r") as f:
                x0 = f["x"][:].astype(np.uint8)
                y0 = f["y"][:].astype(np.float32)
            x = np.concatenate([x, x0], axis=0)
            y = np.concatenate([y, y0], axis=0)
    return x, y
