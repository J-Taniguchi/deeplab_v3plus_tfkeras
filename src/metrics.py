import tensorflow as tf

def IoU(y_true, y_pred):
    p = tf.dtypes.cast(y_pred > 0.5, tf.float32)
    Intersection = tf.dtypes.cast(tf.math.equal  ((y_true * p), 1.0) , tf.float32)
    union        = tf.dtypes.cast(tf.math.greater_equal((y_true + p), 1.0) , tf.float32)

    Intersection = tf.reduce_sum(Intersection)
    union        = tf.reduce_sum(union)
    iou = Intersection /(union + tf.keras.backend.epsilon())
    return iou
