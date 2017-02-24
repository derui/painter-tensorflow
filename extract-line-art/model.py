# coding: utf-8

import tensorflow as tf
import operations as op


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

    def __init__(self,
                 out_ch,
                 patch_w,
                 patch_h,
                 activation=tf.nn.relu,
                 name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_ch = out_ch
        self.activation = activation
        self.batch_norm = op.BatchNormalization(name=name)

    def encode(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, input_shape[2], self.out_ch],
            name="weight")
        bias = bias_variable([self.out_ch], name='bias')
        conv = tf.nn.conv2d(
            tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
        conv = self.activation(self.batch_norm.batch_normalization(conv))

        return conv


class MaxPool(object):
    def __call__(self, conv):
        pool = tf.nn.max_pool(
            conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 out_ch,
                 patch_w,
                 patch_h,
                 activation=tf.nn.relu,
                 padding='SAME',
                 name='decoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_ch = out_ch
        self.activation = activation
        self.padding = padding
        self.batch_norm = op.BatchNormalization(name=name)

    def decode(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.out_ch, input_shape[2]],
            name='weight')

        bias = bias_variable([self.out_ch], name="bias")

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight, [1, input_shape[0] * 2, input_shape[1] * 2, self.out_ch],
            [1, 2, 2, 1],
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
        conv = self.activation(self.batch_norm.batch_normalization(conv))

        tf.summary.image('conv', tf.slice(conv, [0, 0, 0, 0], [-1, -1, -1, 1]))
        tf.summary.image('conv2', tf.slice(conv, [0, 0, 0, 1],
                                           [-1, -1, -1, 1]))
        tf.summary.image('conv3', tf.slice(conv, [0, 0, 0, 2],
                                           [-1, -1, -1, 1]))

        return conv


def generator(image, width, height, channels):
    """Make construction layer.
    """

    input_shape = (width, height, channels)

    with tf.name_scope('encoder1'):
        conv1 = Encoder(64, 5, 5, name='encoder1').encode(image, input_shape)
        pool1 = MaxPool()(conv1)
    with tf.name_scope('encoder2'):
        conv2 = Encoder(
            128, 5, 5, name='encoder2').encode(pool1,
                                               [width // 2, height // 2, 64])
        pool2 = MaxPool()(conv2)
    with tf.name_scope('encoder3'):
        conv3 = Encoder(
            256, 5, 5, name='encoder3').encode(pool2,
                                               [width // 4, height // 4, 128])
        pool3 = MaxPool()(conv3)

    with tf.variable_scope('decoder1'):
        deconv1 = Decoder(
            128, 5, 5, name='decoder1').decode(pool3,
                                               [width // 8, height // 8, 256])
    with tf.variable_scope('decoder2'):
        deconv2 = Decoder(
            64, 5, 5, name='decoder2').decode(deconv1,
                                              [width // 4, height // 4, 128])
    with tf.variable_scope('decoder3'):
        deconv3 = Decoder(
            channels, 5, 5, name='decoder3', activation=tf.nn.tanh).decode(
                deconv2, [width // 2, height // 2, 64])

    return deconv3


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
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

    return train_step
