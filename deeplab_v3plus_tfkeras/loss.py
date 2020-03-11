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
        calculate loss about center of the object.
        Args:
            y_true :
            y_pred :
        Return:
            float
        """
        #y_true and y_pred are about center.
        pos_mask = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
        neg_mask = tf.dtypes.cast(tf.math.less (y_true, 1.0), tf.float32)

        pos_loss = ((    y_pred) ** gamma) * tf.math.log(tf.clip_by_value(    y_pred, 1e-4, 1.0))
        neg_loss = ((1 - y_pred) ** gamma) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1.0))

        pos_loss = -1.0 * pos_loss * pos_mask * alpha
        neg_loss = -1.0 * neg_loss * neg_mask * (1 - alpha)

        pos_loss = tf.math.reduce_sum(pos_loss)
        neg_loss = tf.math.reduce_sum(neg_loss)

        return pos_loss + neg_loss

    return focal_loss

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
