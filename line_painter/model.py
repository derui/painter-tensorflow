# coding: utf-8

import tensorflow as tf
from . import operations as op


# 指定したShapeでWeight Variableを作成する。
def weight_variable(shape, name=None):
    # stddevは標準偏差。truncated_normalは、指定した平均（デフォルト０）
    # と、渡した標準偏差（デフォルト１）から、標準偏差の二倍以上の値
    # をtruncateして再度取得するようにする
    return tf.get_variable(
        name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0.1))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 patch_w,
                 patch_h,
                 name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.in_ch, self.out_ch],
            name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(
            tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

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
                 in_ch,
                 out_ch,
                 patch_w,
                 patch_h,
                 padding='SAME',
                 name='decoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = padding
        self.name = name

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.out_ch, self.in_ch],
            name='{}_weight'.format(self.name))

        bias = bias_variable([self.out_ch], name="{}_bias".format(self.name))

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight, [
                tf.shape(tensor)[0], input_shape[0] * 2, input_shape[1] * 2,
                self.out_ch
            ], [1, 2, 2, 1],
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Generator(object):
    def __init__(self):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')
        self.bnc5 = op.BatchNormalization(name='bnc5')
        self.bnc6 = op.BatchNormalization(name='bnc6')
        self.bnc7 = op.BatchNormalization(name='bnc7')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd5 = op.BatchNormalization(name='bnd5')
        self.bnd6 = op.BatchNormalization(name='bnd6')
        
        self.conv1 = Encoder(3, 12, 5, 5, name='encoder1')
        self.conv2 = Encoder(12, 32, 5, 5, name='encoder2')
        self.conv3 = Encoder(32, 64, 5, 5, name='encoder3')
        self.conv4 = Encoder(64, 128, 5, 5, name='encoder4')
        self.conv5 = Encoder(128, 256, 5, 5, name='encoder5')
        self.conv6 = Encoder(256, 512, 5, 5, name='encoder6')
        self.conv7 = Encoder(512, 512, 5, 5, name='encoder7')

        self.pool1 = MaxPool()
        self.pool2 = MaxPool()
        self.pool3 = MaxPool()
        self.pool4 = MaxPool()
        self.pool5 = MaxPool()
        self.pool6 = MaxPool()

        self.deconv1 = Decoder(512, 256, 5, 5, name='decoder1')
        self.deconv2 = Decoder(512, 128, 5, 5, name='decoder2')
        self.deconv3 = Decoder(256, 64, 5, 5, name='decoder3')
        self.deconv4 = Decoder(128, 32, 5, 5, name='decoder4')
        self.deconv5 = Decoder(64, 12, 5, 5, name='decoder5')
        self.deconv6 = Encoder(24, 3, 5, 5, name='decoder6')


def generator(image, width, height, channels):
    """Make construction layer.
    """

    gen = Generator()

    relu = tf.nn.relu
    tanh = tf.nn.tanh

    conv1 = relu(gen.bnc1(gen.conv1(image, [width, height])))
    conv2 = relu(gen.bnc2(gen.conv2(gen.pool1(conv1), [width // 2, height // 2])))
    conv3 = relu(gen.bnc3(gen.conv3(gen.pool2(conv2), [width // 4, height // 4])))
    conv4 = relu(gen.bnc4(gen.conv4(gen.pool3(conv3), [width // 8, height // 8])))
    conv5 = relu(gen.bnc5(gen.conv5(gen.pool4(conv4), [width // 16, height // 16])))
    conv6 = relu(gen.bnc6(gen.conv6(gen.pool5(conv5), [width // 32, height // 32])))

    deconv1 = relu(gen.bnd1(gen.deconv1(conv6, [width // 32, height // 32])))
    deconv2 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv1, conv5], 3), [width // 16, height // 16])))
    deconv3 = relu(gen.bnd3(gen.deconv3(tf.concat([deconv2, conv4], 3), [width // 8, height // 8])))
    deconv4 = relu(gen.bnd4(gen.deconv4(tf.concat([deconv3, conv3], 3), [width // 4, height // 4])))
    deconv5 = relu(gen.bnd5(gen.deconv5(tf.concat([deconv4, conv2], 3), [width // 2, height // 2])))
    deconv6 = tanh(gen.bnd6(gen.deconv6(tf.concat([deconv5, conv1], 3), [width, height])))

    return deconv6


def loss(original_image, output_image, x):
    with tf.name_scope('optimizer'):

        sqrt = tf.square(original_image - output_image)
        tf.summary.image('input', x)
        tf.summary.image('output', output_image)
        tf.summary.image('origin', original_image)
        cross_entropy = tf.reduce_mean(sqrt)
        tf.summary.scalar('entropy', cross_entropy)
    return cross_entropy


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(
            loss, global_step=global_step, var_list=var_list)

    return train_step
