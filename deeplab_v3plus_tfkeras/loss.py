import tensorflow as tf


def make_overwrap_crossentropy(n_labels):
    def overwrap_crossentropy(y_true, y_pred):
        for i in range(n_labels):
            if i == 0:
                loss = tf.keras.losses.binary_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], from_logits=True)
            else:
                loss += tf.keras.losses.binary_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], from_logits=True)
        return loss / n_labels
    return overwrap_crossentropy


def make_weighted_overwrap_crossentropy(n_labels, weights):
    def weighted_overwrap_crossentropy(y_true, y_pred):
        for i in range(n_labels):
            if i == 0:
                loss = weighted_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], weights[i])
            else:
                loss += weighted_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], weights[i])
        return loss / n_labels
    return weighted_overwrap_crossentropy


def make_overwrap_focalloss(n_labels, alphas, gammas):
    FL = []
    for i in range(n_labels):
        FL.append(make_focal_loss(alphas[i], gammas[i]))

    def overwrap_focalloss(y_true, y_pred):
        for i in range(n_labels):
            if i == 0:
                loss = FL[i](y_true[:,:,:,i], y_pred[:,:,:,i])
            else:
                loss += FL[i](y_true[:,:,:,i], y_pred[:,:,:,i])
        return loss / n_labels

    return overwrap_focalloss


def make_focal_loss(alpha, gamma):
    def focal_loss(y_true, y_pred):
        """
        Args:
            y_true :
            y_pred :
        Return:
            float
        """
        pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
        neg_mask = tf.dtypes.cast(tf.math.less (y_true, 1.0), tf.float32)

        pos_loss = ((1 - y_pred) ** gamma) * tf.math.log(tf.clip_by_value(    y_pred, 1e-4, 1.0))
        neg_loss = ((    y_pred) ** gamma) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0))

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

    return focal_loss


def weighted_crossentropy(y_true, y_pred, weights):
    pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
    neg_mask = tf.dtypes.cast(tf.math.less (y_true, 1.0), tf.float32)

    pos_loss = tf.math.log(tf.clip_by_value(    y_pred, 1e-4, 1.0))
    neg_loss = tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0))

    pos_loss = -1.0 * pos_loss * pos_mask * weights
    neg_loss = -1.0 * neg_loss * neg_mask * (1 - weights)

    return pos_loss + neg_loss


def generalized_dice_loss(y_true, y_pred):
    # from https://arxiv.org/pdf/1707.03237.pdf
    # "Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations"
    epsilon = 1e-6

    n_pos = tf.reduce_sum(y_true)
    n_neg = tf.reduce_sum(1 - y_true)

    w1 = 1 / (n_pos**2 + epsilon)
    w2 = 1 / (n_neg**2 + epsilon)

    #_w1 = 1 / (n_pos**2 + epsilon)
    #_w2 = 1 / (n_neg**2 + epsilon)

    #w1 = _w1 / (_w1 + _w2)
    #w2 = _w2 / (_w1 + _w2)

    numerator1 = 2 * tf.reduce_sum(y_true * y_pred, axis=-1) + epsilon
    denominator1 = tf.reduce_sum(y_true + y_pred, axis=-1) + epsilon
    DS1 = numerator1 / denominator1

    numerator2 = 2 * tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=-1) + epsilon
    denominator2 = tf.reduce_sum(2 - y_true - y_pred, axis=-1) + epsilon
    DS2 = numerator2 / denominator2

    return 1 - w1 * DS1 - w2 * DS2

def make_weighted_MSE(image_size):
    pix_count = image_size[0] * image_size[1]
    def weighted_MSE(y_true, y_pred):
        weight = tf.dtypes.cast(tf.math.equal(y_true, 1.0) , tf.float32)
        weight = tf.math.reduce_sum(weight, axis=[0,1,2])
        weight = tf.math.divide(pix_count - weight , (weight+1e-6))
        weight_mat = y_true * weight + 1.0
        loss = tf.math.reduce_mean(((y_true - y_pred)**2)*weight_mat, axis=[0])
        loss = loss * y_true * weight
        loss = tf.math.reduce_mean(loss * weight)

        return loss

    return weighted_MSE
