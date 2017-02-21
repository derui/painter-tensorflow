# coding: utf-8

import tensorflow as tf


# 指定したShapeでWeight Variableを作成する。
def weight_variable(shape, name=None):
    # stddevは標準偏差。truncated_normalは、指定した平均（デフォルト０）
    # と、渡した標準偏差（デフォルト１）から、標準偏差の二倍以上の値
    # をtruncateして再度取得するようにする
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, out_ch, patch_w, patch_h, activation=tf.nn.relu):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_ch = out_ch
        self.activation = activation

    def encode(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, input_shape[2], self.out_ch],
            name="weight")
        bias = bias_variable([self.out_ch], name='bias')
        conv = self.activation(
            tf.nn.conv2d(tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
            + bias)

        tf.summary.histogram('encode_conv', conv)
        return tf.nn.max_pool(
            conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, out_ch, patch_w, patch_h, activation=tf.nn.relu):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_ch = out_ch
        self.activation = activation

    def decode(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.out_ch, input_shape[2]],
            name='weight')

        bias = bias_variable([self.out_ch], name="bias")

        conv = self.activation(
            tf.nn.conv2d_transpose(
                tensor,
                weight, [
                    tf.shape(tensor)[0], input_shape[0] * 2, input_shape[1] *
                    2, self.out_ch
                ],
                strides=[1, 2, 2, 1],
                padding='SAME') + bias)

        tf.summary.histogram('decode_conv', conv)
        return conv


class FeatureMerger(object):
    """ providing feature merge tensors having same shape. """

    def __init__(self, patch_w, patch_h):
        self.patch_w = patch_w
        self.patch_h = patch_h

    def merge(self, conv, deconv, shape):
        """merge features which are given from same layer of convolution and
        deconvolution, and execute convolution to original shape
        """
        concatenated = tf.concat([conv, deconv], 3)

        weight = weight_variable(
            [self.patch_w, self.patch_h, shape[2] * 2, shape[2]],
            name="weight")
        bias = bias_variable([shape[2]], name='bias')
        return tf.nn.relu(
            tf.nn.conv2d(
                concatenated, weight, strides=[1, 1, 1, 1], padding='SAME') +
            bias)


def construction(image, width, height, channels):
    """Make construction layer.
    """

    input_shape = (width, height, channels)

    with tf.name_scope('encoder1'):
        conv1 = Encoder(64, 5, 5).encode(image, input_shape)
    with tf.name_scope('encoder2'):
        conv2 = Encoder(128, 5, 5).encode(conv1, [width // 2, height // 2, 64])
    with tf.name_scope('encoder3'):
        conv3 = Encoder(256, 5, 5).encode(conv2,
                                          [width // 4, height // 4, 128])
    with tf.name_scope('encoder4'):
        conv4 = Encoder(512, 5, 5).encode(conv3,
                                          [width // 8, height // 8, 256])

    with tf.name_scope('encoder5'):
        conv5 = Encoder(1024, 5, 5).encode(conv4,
                                           [width // 16, height // 16, 512])

    with tf.name_scope('decoder1'):
        deconv1 = Decoder(512, 5, 5).decode(conv5,
                                            [width // 32, height // 32, 1024])
        merged1 = FeatureMerger(5, 5).merge(conv4, deconv1,
                                            [width // 16, height // 16, 512])
    with tf.name_scope('decoder2'):
        deconv2 = Decoder(256, 5, 5).decode(merged1,
                                            [width // 16, height // 16, 512])
        merged2 = FeatureMerger(5, 5).merge(conv3, deconv2,
                                            [width // 8, height // 8, 256])
    with tf.name_scope('decoder3'):
        deconv3 = Decoder(128, 5, 5).decode(merged2,
                                            [width // 8, height // 8, 256])
        merged3 = FeatureMerger(5, 5).merge(conv2, deconv3,
                                            [width // 4, height // 4, 128])

    with tf.name_scope('decoder4'):
        deconv4 = Decoder(64, 5, 5).decode(merged3,
                                           [width // 4, height // 4, 128])
        merged4 = FeatureMerger(5, 5).merge(conv1, deconv4,
                                            [width // 2, height // 2, 64])

    with tf.name_scope('decoder4'):
        deconv5 = Decoder(
            channels, 5, 5, activation=tf.nn.sigmoid).decode(
                merged4, [width // 2, height // 2, 64])

    return deconv5


def loss(original_image, output_image, x):
    with tf.name_scope('optimizer'):

        sqrt = tf.square(original_image - output_image)
        tf.summary.image('input', x)
        tf.summary.image('output', output_image)
        tf.summary.image('origin', original_image)
        cross_entropy = tf.reduce_mean(sqrt)
        tf.summary.scalar('entropy', cross_entropy)
    return cross_entropy


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

    return train_step
