import tensorflow as tf
# import numpy as np


def augmentor(image, mask, extra_x=None):
    # you can edit here for image augmentation.
    # image and mask are 3 dim tensors.
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + 0.5), tf.bool)
    if extra_x is None:
        image, mask = tf.cond(
            should_apply_op,
            lambda: (tf.image.flip_left_right(image),
                     tf.image.flip_left_right(mask)),
            lambda: (image, mask))
    else:
        image, mask, extra_x = tf.cond(
            should_apply_op,
            lambda: (tf.image.flip_left_right(image),
                     tf.image.flip_left_right(mask),
                     tf.image.flip_left_right(extra_x)),
            lambda: (image, mask, extra_x))

    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + 0.5), tf.bool)

    if extra_x is None:
        image, mask = tf.cond(
            should_apply_op,
            lambda: (tf.image.flip_up_down(image),
                     tf.image.flip_up_down(mask)),
            lambda: (image, mask))
    else:
        image, mask, extra_x = tf.cond(
            should_apply_op,
            lambda: (tf.image.flip_up_down(image),
                     tf.image.flip_up_down(mask),
                     tf.image.flip_up_down(extra_x)),
            lambda: (image, mask, extra_x))

    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_brightness(image, 0.1)
    # image = tf.image.random_jpeg_quality(image, 90, 100)
    if extra_x is None:
        return image, mask
    else:
        return image, mask, extra_x


def data_augment(image, mask, image_size, p, extra_x=None):
    # if p < 0.0:
    #    raise ValueError("p < 0. p must be positive number.")
    # if len(image.shape) != 3:
    #    raise Exception("dimension of images for data_augment must be 3")
    # if len(mask.shape) != 3:
    #    raise Exception("dimension of masks for data_augment must be 3")
    p = float(p)
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + p), tf.bool)
    if extra_x is None:
        image, mask = tf.cond(
            should_apply_op,
            lambda: augmentor(image, mask),
            lambda: (image, mask))

        return image, mask
    else:
        image, mask, extra_x = tf.cond(
            should_apply_op,
            lambda: augmentor(image, mask, extra_x),
            lambda: (image, mask, extra_x))

        return image, mask, extra_x
