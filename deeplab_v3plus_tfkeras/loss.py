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

def make_overwrap_focalloss(n_labels):
    def overwrap_focalloss(y_true, y_pred):
        for i in range(n_labels):
            if i == 0:
                loss = focal_loss(y_true[:,:,:,i], y_pred[:,:,:,i])
            else:
                loss += focal_loss(y_true[:,:,:,i], y_pred[:,:,:,i])
        return loss / n_labels
    return overwrap_focalloss

@tf.function
def focal_loss(y_true, y_pred):
    """
    calculate loss about center of the object.
    Args:
        y_true :
        y_pred :
    Return:
        float
    """
    #y_true and y_pred are about center.
    alpha = 2
    beta  = 4
    pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
    neg_mask = tf.dtypes.cast(tf.math.less (y_true, 1.0), tf.float32)

    pos_loss = tf.math.pow(1 - y_pred, alpha) * tf.math.log(tf.clip_by_value(    y_pred, 1e-4, 1.0))
    neg_loss = tf.math.pow(1 - y_true, beta ) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0)) * tf.math.pow(y_pred, alpha)

    pos_loss = -1.0 * pos_loss * pos_mask
    neg_loss = -1.0 * neg_loss * neg_mask

    num_pos  = tf.math.reduce_sum(pos_mask)
    pos_loss = tf.math.reduce_sum(pos_loss)
    neg_loss = tf.math.reduce_sum(neg_loss)

    #return (pos_loss + neg_loss) / (num_pos + 1e-4)
    #cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    if num_pos <= 0:
        return neg_loss
    else:
        return (pos_loss + neg_loss) / num_pos

def weighted_crossentropy(y_true, y_pred, weights):
    pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
    neg_mask = tf.dtypes.cast(tf.math.less (y_true, 1.0), tf.float32)

    pos_loss = tf.math.log(tf.clip_by_value(    y_pred, 1e-4, 1.0))
    neg_loss = tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0))

    pos_loss = -1.0 * pos_loss * pos_mask * weights[0]
    neg_loss = -1.0 * neg_loss * neg_mask * weights[1]

    return pos_loss + neg_loss


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
