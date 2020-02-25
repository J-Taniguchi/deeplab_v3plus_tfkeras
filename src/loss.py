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
                loss = tf.keras.losses.binary_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], from_logits=True) * weights[i]
            else:
                loss += tf.keras.losses.binary_crossentropy(y_true[:,:,:,i], y_pred[:,:,:,i], from_logits=True) * weights[i]
        return loss / n_labels
    return weighted_overwrap_crossentropy

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
