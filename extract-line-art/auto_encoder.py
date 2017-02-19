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

    def encode(self, tensor):
        shape = tensor.get_shape()
        weight = weight_variable(
            [self.patch_w, self.patch_h, shape[3].value, self.out_ch],
            name="weight")
        bias = bias_variable([self.out_ch], name='bias')
        conv = self.activation(
            tf.nn.conv2d(tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
            + bias)

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

    def decode(self, tensor):
        shape = tf.shape(tensor)
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.out_ch, self.out_ch * 4],
            name='weight')
        bias = bias_variable([self.out_ch], name="bias")

        return self.activation(
            tf.nn.conv2d_transpose(
                tensor,
                weight,
                [tf.shape(tensor)[0], shape[1] * 2, shape[2] * 2, self.out_ch],
                strides=[1, 2, 2, 1],
                padding='SAME') + bias)


def construction(image, width, height, channels):
    """Make construction layer.
    """

    with tf.name_scope('encoder1'):
        conv1 = Encoder(channels * 4, 5, 5).encode(image)
    with tf.name_scope('encoder2'):
        conv2 = Encoder(channels * 16, 5, 5).encode(conv1)
    with tf.name_scope('decoder1'):
        deconv1 = Decoder(channels * 4, 5, 5).decode(conv2)
    with tf.name_scope('decoder2'):
        deconv2 = Decoder(
            channels, 5, 5, activation=tf.nn.sigmoid).decode(deconv1)

    return deconv2


def loss(original_image, output_image):
    with tf.name_scope('optimizer'):
        _, width, height, channels = original_image.get_shape()
        shape = [-1, int(width * height * channels)]
        l_orig = tf.reshape(original_image, shape)
        l_out = tf.reshape(output_image, shape)

        logloss = l_orig * tf.log(l_out) + (tf.subtract(
            1.0, l_orig)) * tf.log(tf.subtract(1.0, l_out))
        tf.summary.image('output', output_image)
        tf.summary.image('origin', original_image)
        tf.summary.image('entropy',
                         tf.reshape(logloss, tf.shape(original_image)))
        cross_entropy = -tf.reduce_mean(logloss)
        tf.summary.scalar('entropy', cross_entropy)
    return cross_entropy


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

    return train_step
