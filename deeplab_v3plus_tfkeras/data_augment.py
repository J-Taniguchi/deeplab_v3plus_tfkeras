import tensorflow as tf
import numpy as np

def augmentor(image, mask):
    # you can edit here for image augmentation.
    # image and mask are 3 dim tensors.
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + 0.5), tf.bool)
    image, mask = tf.cond(
        should_apply_op,
        lambda: (tf.image.flip_left_right(image),
                 tf.image.flip_left_right(mask)),
        lambda: (image, mask))
    image = tf.image.flip_left_right(image)
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_saturation(image, 0.3, 1.0)
    image = tf.image.random_jpeg_quality(image, 90, 100)

    return image, mask

def data_augment(image, mask, image_size, p):
    #if p < 0.0:
    #    raise ValueError("p < 0. p must be positive number.")
    #if len(image.shape) != 3:
        #raise Exception("dimension of images for data_augment must be 3")
    #if len(mask.shape) != 3:
        #raise Exception("dimension of masks for data_augment must be 3")
    p = float(p)
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + p), tf.bool)
    image, mask = tf.cond(
        should_apply_op,
        lambda: augmentor(image, mask),
        lambda: (image, mask))

    return image, mask
