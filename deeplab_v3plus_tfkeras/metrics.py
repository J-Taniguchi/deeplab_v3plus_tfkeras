import tensorflow as tf

epsilon = tf.keras.backend.epsilon()

def make_IoU(threshold=0.5):
    def IoU(y_true, y_pred):
        p = tf.dtypes.cast(y_pred > threshold, tf.float32)
        Intersection = tf.dtypes.cast(tf.math.equal  ((y_true * p), 1.0) , tf.float32)
        union        = tf.dtypes.cast(tf.math.greater_equal((y_true + p), 1.0) , tf.float32)

        Intersection = tf.reduce_sum(Intersection)
        union        = tf.reduce_sum(union)

        if union.numpy() == 0:
            iou = 1.0
        else:
            iou = Intersection / union
        return iou
    return IoU
