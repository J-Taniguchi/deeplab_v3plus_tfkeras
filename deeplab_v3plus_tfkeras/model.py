import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


def deeplab_v3plus(image_size, n_categories):
    if np.mod(image_size[0], 32) != 0 or np.mod(image_size[1], 32) != 0:
        raise Exception("image_size must be multiples of 32")

    if min(image_size) < 320:
        raise Exception("minimum(image_size) must be larger or equal than 320")

    # xm means x_main. center flow of the fig. 4.
    # xs means x_side. side flow of the fig. 4.
    # encoder
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3), name="inputs")
    # entry_flow
    # entry block 1
    xm = Conv_BN(inputs, 32, filter=3, prefix="entry_b1", suffix="1", strides=2, dilation_rate=1)
    xm = xs = Conv_BN(xm, 64, filter=3, prefix="entry_b1", suffix="2", strides=1, dilation_rate=1)

    # entry block 2
    # xm = layers.DepthwiseConv2D((3,3), depth_multiplier=2, padding="same", name="entry_b2_dcv1")(xm)
    n_channels = 128
    xm = SepConv_BN(xm, n_channels, prefix="entry_b2", suffix="1", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b2", suffix="2", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b2", suffix="3", strides=2, dilation_rate=1)
    xs = Conv_BN(xs, n_channels, filter=1, prefix="entry_b2_side", suffix="1", strides=2, dilation_rate=1)
    xs = xm = layers.add([xs, xm], name="entry_b2_add")
    # entry block 3
    n_channels = 256
    xm = SepConv_BN(xm, n_channels, prefix="entry_b3", suffix="1", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b3", suffix="2", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b3", suffix="3", strides=2, dilation_rate=1)
    xs = Conv_BN(xs, n_channels, filter=1, prefix="entry_b3_side", suffix="1", strides=2, dilation_rate=1)
    xs = xm = x_dec = layers.add([xs, xm], name="entry_b3_add")
    # entry block 4
    n_channels = 728
    xm = SepConv_BN(xm, n_channels, prefix="entry_b4", suffix="1", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b4", suffix="2", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, n_channels, prefix="entry_b4", suffix="3", strides=2, dilation_rate=1)
    xs = Conv_BN(xs, n_channels, filter=1, prefix="entry_b4_side", suffix="1", strides=2, dilation_rate=1)
    xs = xm = layers.add([xs, xm], name="entry_b4_add")  # middle flow
    for i in range(16):
        ii = i + 1
        xm = SepConv_BN(xm, n_channels, prefix="middle_b%d" % ii, suffix="1", strides=1, dilation_rate=1)
        xm = SepConv_BN(xm, n_channels, prefix="middle_b%d" % ii, suffix="2", strides=1, dilation_rate=1)
        xm = SepConv_BN(xm, n_channels, prefix="middle_b%d" % ii, suffix="3", strides=1, dilation_rate=1)
        xs = xm = layers.add([xs, xm], name="middle_b%d_add" % ii)  # middle flow
    # exit flow
    # exit block1
    xm = SepConv_BN(xm, 728, prefix="exit_b1", suffix="1", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, 1024, prefix="exit_b1", suffix="2", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, 1024, prefix="exit_b1", suffix="3", strides=2, dilation_rate=1)
    xs = Conv_BN(xs, 1024, filter=1, prefix="exit_b1_side", suffix="1", strides=2, dilation_rate=1)
    xs = xm = layers.add([xs, xm], name="exit_b1_add")  # middle flow

    # exit block2
    xm = SepConv_BN(xm, 1536, prefix="exit_b2", suffix="1", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, 1536, prefix="exit_b2", suffix="2", strides=1, dilation_rate=1)
    xm = SepConv_BN(xm, 2048, prefix="exit_b2", suffix="3", strides=1, dilation_rate=1)

    # encoder = keras.Model(inputs=inputs,outputs=xm, name="xception_encoder")

    # get feature_size and cal dilation_rates
    feature_size = keras.backend.int_shape(xm)[1:3]
    min_feature_size = min(feature_size)
    dilation_rates = cal_dilation_rates(min_feature_size)

    # ASPP
    aspp1 = Conv_BN(xm, 256, filter=1, prefix="aspp1", suffix="1", strides=1, dilation_rate=1)
    aspp2 = SepConv_BN(xm, 256, prefix="aspp2", suffix="1", strides=1, dilation_rate=dilation_rates[0])
    aspp3 = SepConv_BN(xm, 256, prefix="aspp3", suffix="1", strides=1, dilation_rate=dilation_rates[1])
    aspp4 = SepConv_BN(xm, 256, prefix="aspp4", suffix="1", strides=1, dilation_rate=dilation_rates[2])

    aspp5 = keras.backend.mean(xm, axis=[1, 2], keepdims=True)
    aspp5 = Conv_BN(aspp5, 256, filter=1, prefix="aspp5", suffix="1", strides=1, dilation_rate=1)
    aspp5 = layers.UpSampling2D(feature_size, name="aspp5_upsampling")(aspp5)

    ASPP = layers.concatenate([aspp1, aspp2, aspp3, aspp4, aspp5], name="ASPP")
    ASPP = Conv_BN(ASPP, 256, filter=1, prefix="ASPP", suffix="1", strides=1, dilation_rate=1)
    ASPP = layers.UpSampling2D(4, name="ASPP_upsample_4")(ASPP)

    # decoder
    x_dec = Conv_BN(x_dec, 48, filter=1, prefix="dec1", suffix="1", strides=1, dilation_rate=1)
    x_dec = layers.concatenate([x_dec, ASPP], name="dec_concat")
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="2", strides=1, dilation_rate=1)
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="3", strides=1, dilation_rate=1)
    x_dec = layers.UpSampling2D(4, name="dec_upsample_1")(x_dec)

    x_dec = SepConv_BN(x_dec, n_categories, prefix="dec2", suffix="1", strides=1, dilation_rate=1, last_activation=False)
    x_dec = layers.UpSampling2D(2, name="dec_upsample_2")(x_dec)
    outputs = layers.Activation(tf.nn.softmax, name="softmax")(x_dec)

    model = keras.Model(inputs=inputs, outputs=outputs, name="deeplab-v3plus")
    return model


def deeplab_v3plus_transfer_os16(n_categories,
                                 encoder,
                                 layer_name_to_decoder,
                                 encoder_end_layer_name,
                                 freeze_encoder=True,
                                 output_activation='softmax',
                                 batch_renorm=False):

    layer_dict = dict([(layer.name, layer) for layer in encoder.layers])
    inputs = encoder.input
    xm = layer_dict[encoder_end_layer_name].output
    x_dec = layer_dict[layer_name_to_decoder].output
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    feature_size = keras.backend.int_shape(xm)[1:3]
    min_feature_size = min(feature_size)
    dilation_rates = cal_dilation_rates(min_feature_size)

    # ASPP
    aspp1 = Conv_BN(xm, 256, filter=1, prefix="aspp1", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    aspp2 = SepConv_BN(xm, 256, prefix="aspp2", suffix="1", strides=1, dilation_rate=dilation_rates[0], batch_renorm=batch_renorm)
    aspp3 = SepConv_BN(xm, 256, prefix="aspp3", suffix="1", strides=1, dilation_rate=dilation_rates[1], batch_renorm=batch_renorm)
    aspp4 = SepConv_BN(xm, 256, prefix="aspp4", suffix="1", strides=1, dilation_rate=dilation_rates[2], batch_renorm=batch_renorm)

    aspp5 = keras.backend.mean(xm, axis=[1, 2], keepdims=True)
    aspp5 = Conv_BN(aspp5, 256, filter=1, prefix="aspp5", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    aspp5 = layers.UpSampling2D(feature_size, name="aspp5_upsampling")(aspp5)

    ASPP = layers.concatenate([aspp1, aspp2, aspp3, aspp4, aspp5], name="ASPP")
    ASPP = Conv_BN(ASPP, 256, filter=1, prefix="ASPP", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    ASPP = layers.UpSampling2D(4, name="ASPP_upsample_4")(ASPP)

    # decoder
    x_dec = Conv_BN(x_dec, 48, filter=1, prefix="dec1", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    # print("in decoder, layer from encoder is resized from " + str(x_dec.shape[1:3]) + " to " + str(ASPP.shape[1:3]))
    # x_dec = Resize_Layer(x_dec, ASPP.shape[1:3], name="dec1_resize")
    x_dec = layers.concatenate([x_dec, ASPP], name="dec_concat")
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="2", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="3", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = layers.UpSampling2D(4, name="dec_upsample_2")(x_dec)
    x_dec = SepConv_BN(x_dec, n_categories, prefix="dec2", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)

    # x_dec = SepConv_BN(x_dec, n_categories, prefix="dec2", suffix="1", strides=1, dilation_rate=1, last_activation=False)
    # x_dec = layers.UpSampling2D(2, name="dec_upsample_3")(x_dec)
    # x_dec = Conv_BN(x_dec, n_categories, prefix="last_dec", suffix="1", strides=1, dilation_rate=1)

    if output_activation == 'softmax':
        outputs = layers.Activation(tf.nn.softmax, name="softmax")(x_dec)
    elif output_activation == 'sigmoid':
        outputs = layers.Activation(tf.nn.sigmoid, name="sigmoid")(x_dec)

    model = keras.Model(inputs=inputs, outputs=outputs, name=encoder.name + "_deeplab-v3plus")
    return model


def deeplab_v3plus_transfer_extra_channels(n_categories,
                                           encoder,
                                           layer_name_to_decoder,
                                           encoder_end_layer_name,
                                           n_extra_channels=1,
                                           freeze_encoder=True,
                                           output_activation='softmax',
                                           batch_renorm=False,):
    # ratio of input image size and extra_x size must be 10:1
    # you can adjast this ratio by modifing extra_x_upsample and shape of input_extra.

    if n_extra_channels < 1:
        raise Exception("n_extra_channels must be more than 1")

    layer_dict = dict([(layer.name, layer) for layer in encoder.layers])
    xm = layer_dict[encoder_end_layer_name].output
    x_dec = layer_dict[layer_name_to_decoder].output
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    feature_size = keras.backend.int_shape(xm)[1:3]
    min_feature_size = min(feature_size)
    dilation_rates = cal_dilation_rates(min_feature_size)

    input_rgb = encoder.input
    input_size = keras.backend.int_shape(input_rgb)[1:3]
    input_extra = tf.keras.Input(shape=(input_size[0] // 10, input_size[1] // 10, n_extra_channels), name="input_extra")

    # encoder(input_rgb)

    # ASPP
    aspp1 = Conv_BN(xm, 256, filter=1, prefix="aspp1", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    aspp2 = SepConv_BN(xm, 256, prefix="aspp2", suffix="1", strides=1, dilation_rate=dilation_rates[0], batch_renorm=batch_renorm)
    aspp3 = SepConv_BN(xm, 256, prefix="aspp3", suffix="1", strides=1, dilation_rate=dilation_rates[1], batch_renorm=batch_renorm)
    aspp4 = SepConv_BN(xm, 256, prefix="aspp4", suffix="1", strides=1, dilation_rate=dilation_rates[2], batch_renorm=batch_renorm)

    aspp5 = keras.backend.mean(xm, axis=[1, 2], keepdims=True)
    aspp5 = Conv_BN(aspp5, 256, filter=1, prefix="aspp5", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    aspp5 = layers.UpSampling2D(feature_size, name="aspp5_upsampling")(aspp5)

    ASPP = layers.concatenate([aspp1, aspp2, aspp3, aspp4, aspp5], name="ASPP")
    ASPP = Conv_BN(ASPP, 256, filter=1, prefix="ASPP", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    ASPP = layers.UpSampling2D(4, name="ASPP_upsample_4")(ASPP)

    # decoder
    x_dec = Conv_BN(x_dec, 48, filter=1, prefix="dec1", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    # print("in decoder, layer from encoder is resized from " + str(x_dec.shape[1:3]) + " to " + str(ASPP.shape[1:3]))
    # x_dec = Resize_Layer(x_dec, ASPP.shape[1:3], name="dec1_resize")
    x_dec = layers.concatenate([x_dec, ASPP], name="dec_concat")
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="2", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = SepConv_BN(x_dec, 256, prefix="dec1", suffix="3", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = layers.UpSampling2D(4, name="dec_upsample_2")(x_dec)

    # extra inpts
    x_extra = SepConv_BN(input_extra, 8, prefix="ext1", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = SepConv_BN(x_extra, 16, prefix="ext1", suffix="2", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = SepConv_BN(x_extra, 32, prefix="ext1", suffix="3", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = layers.UpSampling2D(5, name="extra_x_upsample_1")(x_extra)

    x_extra = SepConv_BN(x_extra, 64, prefix="ext2", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = SepConv_BN(x_extra, 128, prefix="ext2", suffix="2", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = SepConv_BN(x_extra, 256, prefix="ext2", suffix="3", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_extra = layers.UpSampling2D(2, name="extra_x_upsample_2")(x_extra)

    x_dec = layers.concatenate([x_dec, x_extra], name="concat_extra")

    x_dec = SepConv_BN(x_dec, 128, prefix="dec2", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = SepConv_BN(x_dec, 64, prefix="dec2", suffix="2", strides=1, dilation_rate=1, batch_renorm=batch_renorm)
    x_dec = SepConv_BN(x_dec, 32, prefix="dec2", suffix="3", strides=1, dilation_rate=1, batch_renorm=batch_renorm)

    x_dec = SepConv_BN(x_dec, n_categories, prefix="dec_last", suffix="1", strides=1, dilation_rate=1, batch_renorm=batch_renorm)

    if output_activation == 'softmax':
        outputs = layers.Activation(tf.nn.softmax, name="softmax")(x_dec)
    elif output_activation == 'sigmoid':
        outputs = layers.Activation(tf.nn.sigmoid, name="sigmoid")(x_dec)

    model = keras.Model(inputs=(input_rgb, input_extra), outputs=outputs, name=encoder.name + "_deeplab-v3plus")
    return model


def cal_dilation_rates(im_size):
    max_atrous_rate = im_size // 2 - 1
    if max_atrous_rate <= 3:
        raise Exception("minimum(image_size) is too small")
    else:
        a = np.linspace(0, 32 // 3, 4)
        return np.ceil(a).astype(np.int)[1:]


def SepConv_BN(x, n_channels, prefix=" ", suffix=" ", strides=1, dilation_rate=1, last_activation=True, batch_renorm=False):
    x = layers.DepthwiseConv2D((3, 3), strides, dilation_rate=(dilation_rate, dilation_rate), padding="same", name=prefix + "_sc" + suffix)(x)
    x = layers.BatchNormalization(name=prefix + "_sc-bn" + suffix, renorm=batch_renorm)(x)
    x = layers.Activation(tf.nn.relu, name=prefix + "_sc-act" + suffix)(x)
    x = layers.Conv2D(n_channels, (1, 1), padding="same", name=prefix + "_cv" + suffix)(x)
    x = layers.BatchNormalization(name=prefix + "_cv-bn" + suffix, renorm=batch_renorm)(x)
    if last_activation is True:
        x = layers.Activation(tf.nn.relu, name=prefix + "_cv-act" + suffix)(x)
    return x


def Conv_BN(x, n_channels, filter=3, prefix=" ", suffix=" ", strides=1, dilation_rate=1, last_activation=True, batch_renorm=False):
    x = layers.Conv2D(n_channels, filter, strides, dilation_rate=(dilation_rate, dilation_rate), padding="same", name=prefix + "_cv" + suffix)(x)
    x = layers.BatchNormalization(name=prefix + "_bn" + suffix, renorm=batch_renorm)(x)
    if last_activation is True:
        x = layers.Activation(tf.nn.relu, name=prefix + "_act" + suffix)(x)
    return x


def Resize_Layer(inputs, out_tensor_hw, name="resize"):  # resizes input tensor wrt. ref_tensor
    # return tf.image.resize_nearest_neighbor(inputs, out_tensor_hw, name=name)
    return tf.image.resize(inputs, out_tensor_hw, name=name)
