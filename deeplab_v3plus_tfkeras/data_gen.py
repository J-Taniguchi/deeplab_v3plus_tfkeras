# import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
from .data_utils import make_xy_from_data_paths
from .data_augment import data_augment
from .label import Label


def make_array_generator(x,
                         y,
                         preprocess=None,
                         augmentation=True):
    # n_images = x.shape[0]
    # image_size for data_augment is (height, width)
    image_size = x.shape[1:3]

    def map_f(_x, _y):
        # _x = _x[np.newaxis,:,:,:]
        # _y = _y[np.newaxis,:,:,:]

        if augmentation is True:
            # image_size for data_augment is (height, width)
            _x, _y = data_augment(_x,
                                  _y,
                                  image_size=image_size,
                                  p=0.95)
        _x = tf.cast(_x, tf.float32)
        _y = tf.cast(_y, tf.float32)
        if preprocess is None:
            _x = (_x / 127.5) - 1
        else:
            _x = preprocess(_x)
        return _x, _y

    def wrap_mapf(_x, _y):
        x_out, y_out = tf.py_function(map_f,
                                      inp=[_x, _y],
                                      Tout=(tf.float32, tf.float32))
        x_out.set_shape([None, None, None])
        y_out.set_shape([None, None, None])
        return x_out, y_out

    return tf.data.Dataset.from_tensor_slices((x, y)), wrap_mapf


def make_path_generator(img_paths,
                        seg_img_paths,
                        image_size,
                        label: Label,
                        preprocess=None,
                        augmentation=True,
                        resize_or_crop="resize",
                        data_type="image", #or "npy"
                        ):
    def map_f(x_path, y_path):
        x, y = make_xy_from_data_paths(
            [x_path],
            [y_path],
            image_size,
            label,
            resize_or_crop=resize_or_crop,
            data_type=imput_type)
        x = x[0]
        y = y[0]

        if augmentation is True:
            # image_size for data_augment is (height, width)
            x, y = data_augment(x,
                                y,
                                image_size=image_size,
                                p=0.95)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        if preprocess is None:
            x = (x / 127.5) - 1
        else:
            x = preprocess(x)

        return x, y

    def wrap_mapf(x_path, y_path):
        x_out, y_out = tf.py_function(map_f,
                                      inp=[x_path, y_path],
                                      Tout=(tf.float32, tf.float32))
        return x_out, y_out

    ds = tf.data.Dataset.from_tensor_slices((img_paths, seg_img_paths))
    return ds, wrap_mapf
