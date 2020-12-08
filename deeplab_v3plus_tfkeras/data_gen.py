# import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
from .data_utils import make_xy_from_data_path
from .data_augment import data_augment
from .label import Label


def make_generator(img_paths,
                   image_size,
                   label: Label,
                   seg_img_paths=None,
                   preprocess=None,
                   augmentation=True,
                   ):

    def map_f(x_path, y_path=None):
        x, extra_x, y = make_xy_from_data_path(
            x_path,
            y_path,
            image_size,
            label)

        if augmentation is True:
            # image_size for data_augment is (height, width)
            x, y = data_augment(
                x,
                y,
                image_size=image_size,
                p=0.95)

        x = tf.cast(x, tf.float32)

        if y is not None:
            y = tf.cast(y, tf.float32)

        if preprocess is None:
            x = (x / 127.5) - 1
        else:
            x = preprocess(x)

        if y_path is None:
            return x
        else:
            return x, y

    def wrap_mapf(x_path, y_path=None):
        if y_path is None:
            x_out = tf.py_function(
                map_f,
                inp=[x_path],
                Tout=(tf.float32))
            return x_out
        else:
            x_out, y_out = tf.py_function(
                map_f,
                inp=[x_path, y_path],
                Tout=(tf.float32, tf.float32))
            return x_out, y_out

    if seg_img_paths is None:
        ds = tf.data.Dataset.from_tensor_slices((img_paths))
    else:
        ds = tf.data.Dataset.from_tensor_slices((img_paths, seg_img_paths))

    return ds, wrap_mapf


def make_generator_with_extra_x(img_paths,
                                extra_x_paths,
                                image_size,
                                label: Label,
                                seg_img_paths=None,
                                preprocess=None,
                                augmentation=True,
                                ):
    def map_f(x_path, extra_x_path, y_path=None):
        x, extra_x, y = make_xy_from_data_path(
            x_path,
            y_path,
            image_size,
            label,
            extra_x_path=extra_x_path)

        if augmentation is True:
            # image_size for data_augment is (height, width)
            x, y, extra_x = data_augment(
                x,
                y,
                extra_x=extra_x,
                image_size=image_size,
                p=0.95)

        x = tf.cast(x, tf.float32)
        if y is not None:
            y = tf.cast(y, tf.float32)
        if extra_x is not None:
            extra_x = tf.cast(extra_x, tf.float32)

        if preprocess is None:
            x = (x / 127.5) - 1
        else:
            x = preprocess(x)

        if y_path is None:
            return x, extra_x
        else:
            return x, extra_x, y

    if seg_img_paths is None:
        def wrap_mapf(x_path, extra_x_path):
            x_out, extra_x_out = tf.py_function(
                map_f,
                inp=[x_path, extra_x_path],
                Tout=(tf.float32, tf.float32))
            return (x_out, extra_x_out)
        ds = tf.data.Dataset.from_tensor_slices((img_paths, extra_x_paths))
    else:
        def wrap_mapf(x_path, extra_x_path, y_path):
            x_out, extra_x_out, y_out = tf.py_function(
                map_f,
                inp=[x_path, extra_x_path, y_path],
                Tout=(tf.float32, tf.float32, tf.float32))
            return (x_out, extra_x_out), y_out
        ds = tf.data.Dataset.from_tensor_slices((img_paths, extra_x_paths, seg_img_paths))


    return ds, wrap_mapf
