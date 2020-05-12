import tensorflow as tf
import tensorflow.keras as keras
epsilon = keras.backend.epsilon()


def make_focal_loss(n_labels, alpha_list, gamma_list, class_weight=None):
    if class_weight is None:
        class_weight = [1 / n_labels] * n_labels
    FL = []
    for i in range(n_labels):
        FL.append(_make_focal_loss(alpha_list[i], gamma_list[i]))

    def focal_loss(y_true, y_pred):
        for i in range(n_labels):
            if i == 0:
                loss = FL[i](y_true[:, :, :, i], y_pred[:, :, :, i]) * class_weight[i]
            else:
                loss += FL[i](y_true[:, :, :, i], y_pred[:, :, :, i]) * class_weight[i]
        return loss / n_labels

    return focal_loss


def _make_focal_loss(alpha, gamma):
    def _focal_loss(y_true, y_pred):
        """
        Args:
            y_true :
            y_pred :
        Return:
            float
        """
        pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
        neg_mask = tf.dtypes.cast(tf.math.less(y_true, 1.0), tf.float32)

        pos_loss = ((1 - y_pred) ** gamma) * tf.math.log(tf.clip_by_value(y_pred, 1e-4, 1.0))
        neg_loss = (y_pred ** gamma) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0))

        pos_loss = -1.0 * pos_loss * pos_mask * alpha
        neg_loss = -1.0 * neg_loss * neg_mask * (1 - alpha)

        n_pos = tf.math.reduce_sum(pos_mask)
        n_neg = tf.math.reduce_sum(neg_mask)

        if n_pos.numpy() == 0:
            pos_loss = 0.0
        else:
            pos_loss = tf.math.reduce_sum(pos_loss) / n_pos
        if n_neg.numpy() == 0:
            neg_loss = 0.0
        else:
            neg_loss = tf.math.reduce_sum(neg_loss) / n_neg

        return pos_loss + neg_loss

    return _focal_loss


def generalized_dice_loss(y_true, y_pred):
    # from https://arxiv.org/pdf/1707.03237.pdf
    # "Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations"

    n_pos = tf.reduce_sum(y_true)
    n_neg = tf.reduce_sum(1 - y_true)

    # w1 = 1 / (n_pos**2 + epsilon)
    # w2 = 1 / (n_neg**2 + epsilon)

    _w1 = 1 / (n_pos**2 + epsilon)
    _w2 = 1 / (n_neg**2 + epsilon)

    w1 = _w1 / (_w1 + _w2)
    w2 = 1.0 - w1

    numerator1 = 2 * tf.reduce_sum(y_true * y_pred, axis=-1) + epsilon
    denominator1 = tf.reduce_sum(y_true + y_pred, axis=-1) + epsilon
    DS1 = numerator1 / denominator1

    numerator2 = 2 * tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=-1) + epsilon
    denominator2 = tf.reduce_sum(2 - y_true - y_pred, axis=-1) + epsilon
    DS2 = numerator2 / denominator2

    return 1 - w1 * DS1 - w2 * DS2
