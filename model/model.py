# coding: utf-8

import tensorflow as tf
from . import operations as op


# 指定したShapeでWeight Variableを作成する。
def weight_variable(shape, name=None):
    # stddevは標準偏差。truncated_normalは、指定した平均（デフォルト０）
    # と、渡した標準偏差（デフォルト１）から、標準偏差の二倍以上の値
    # をtruncateして再度取得するようにする
    return tf.get_variable(
        name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


def bias_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0.0))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_w, patch_h, strides=[1,1,1,1],name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.in_ch, self.out_ch],
            name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(
            tensor, weight, strides=self.strides, padding='SAME')
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
                 batch_size,
                 padding='SAME',
                 name='decoder'):
        self.batch_size = batch_size
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
                self.batch_size, input_shape[0] * 2, input_shape[1] * 2,
                self.out_ch
            ], [1, 2, 2, 1],
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Generator(object):
    def __init__(self, batch_size):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')
        self.bnc5 = op.BatchNormalization(name='bnc5')
        self.bnc6 = op.BatchNormalization(name='bnc6')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd5 = op.BatchNormalization(name='bnd5')
        self.bnd6 = op.BatchNormalization(name='bnd6')

        self.conv1 = Encoder(1, 24, 5, 5, name='encoder1')
        self.conv2 = Encoder(24, 32, 5, 5, name='encoder2')
        self.conv3 = Encoder(32, 64, 5, 5, name='encoder3')
        self.conv4 = Encoder(64, 128, 5, 5, name='encoder4')
        self.conv5 = Encoder(128, 256, 5, 5, name='encoder5')
        self.conv6 = Encoder(256, 512, 5, 5, name='encoder6')

        self.pool1 = MaxPool()
        self.pool2 = MaxPool()
        self.pool3 = MaxPool()
        self.pool4 = MaxPool()
        self.pool5 = MaxPool()
        self.pool6 = MaxPool()

        self.deconv1 = Decoder(512, 256, 5, 5, batch_size=batch_size, name='decoder1')
        self.deconv2 = Decoder(512, 128, 5, 5, batch_size=batch_size, name='decoder2')
        self.deconv3 = Decoder(256, 64, 5, 5, batch_size=batch_size, name='decoder3')
        self.deconv4 = Decoder(128, 32, 5, 5, batch_size=batch_size, name='decoder4')
        self.deconv5 = Decoder(64, 24, 5, 5, batch_size=batch_size, name='decoder5')
        self.deconv6 = Encoder(24, 3, 1, 1, name='decoder6')


def generator(image, width, height, channels, batch_size):
    """Make construction layer.
    """

    gen = Generator(batch_size)

    relu = tf.nn.relu
    tanh = tf.nn.tanh

    conv1 = relu(gen.bnc1(gen.conv1(image, [width, height])))
    conv2 = relu(
        gen.bnc2(gen.conv2(gen.pool1(conv1), [width // 2, height // 2])))
    conv3 = relu(
        gen.bnc3(gen.conv3(gen.pool2(conv2), [width // 4, height // 4])))
    conv4 = relu(
        gen.bnc4(gen.conv4(gen.pool3(conv3), [width // 8, height // 8])))
    conv5 = relu(
        gen.bnc5(gen.conv5(gen.pool4(conv4), [width // 16, height // 16])))
    conv6 = relu(
        gen.bnc6(gen.conv6(gen.pool5(conv5), [width // 32, height // 32])))

    deconv1 = relu(gen.bnd1(gen.deconv1(conv6, [width // 32, height // 32])))
    deconv2 = relu(
        gen.bnd2(
            gen.deconv2(
                tf.concat([deconv1, conv5], 3), [width // 16, height // 16])))
    deconv3 = relu(
        gen.bnd3(
            gen.deconv3(
                tf.concat([deconv2, conv4], 3), [width // 8, height // 8])))
    deconv4 = relu(
        gen.bnd4(
            gen.deconv4(
                tf.concat([deconv3, conv3], 3), [width // 4, height // 4])))
    deconv5 = relu(
        gen.bnd5(
            gen.deconv5(
                tf.concat([deconv4, conv2], 3), [width // 2, height // 2])))
    deconv6 = tanh(gen.deconv6(deconv5, [width, height]))

    return deconv6


class Discriminator(object):
    """Define discriminator"""

    def __init__(self):
        self.conv1 = Encoder(3, 12, 5, 5, name='encoder1', strides=[1,2,2,1])
        self.conv2 = Encoder(12, 32, 5, 5, name='encoder2', strides=[1,2,2,1])
        self.conv3 = Encoder(32, 64, 5, 5, name='encoder3', strides=[1,2,2,1])
        self.conv4 = Encoder(64, 128, 5, 5, name='encoder4', strides=[1,2,2,1])
        self.conv5 = Encoder(128, 256, 5, 5, name='encoder5', strides=[1,2,2,1])
        self.conv6 = Encoder(256, 512, 5, 5, name='encoder6', strides=[1,2,2,1])

        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')
        self.bnc5 = op.BatchNormalization(name='bnc5')
        self.bnc6 = op.BatchNormalization(name='bnc6')


class LinearEncoder(object):
    """Encoder for Linear Operation."""

    def __init__(self, name='linear_encoder'):
        self.name = name

    def __call__(self, tensor, in_ch, out_ch):
        weight = weight_variable(
            [in_ch, out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([out_ch], name='{}_bias'.format(self.name))
        conv = tf.matmul(tensor, weight)
        conv = tf.nn.bias_add(conv, bias)

        return conv


def discriminator(images, height, width, chan):
    """make discriminator network"""

    D = Discriminator()

    relu = tf.nn.relu
    conv1 = relu(D.bnc1(D.conv1(images, [width, height])))
    conv2 = relu(D.bnc2(D.conv2(conv1, [width // 2, height // 2])))
    conv3 = relu(D.bnc3(D.conv3(conv2, [width // 4, height // 4])))
    conv4 = relu(D.bnc4(D.conv4(conv3, [width // 8, height // 8])))
    conv5 = relu(D.bnc5(D.conv5(conv4, [width // 16, height // 16])))
    conv6 = relu(D.bnc6(D.conv6(conv5, [width // 32, height // 32])))

    w = width // 64
    h = height // 64
    conv6 = tf.reshape(conv6, [-1, w * h * 512])
    logit = LinearEncoder()(conv6, w * h * 512, 1)
    tf.summary.histogram('logit', logit)
    return logit


def d_loss(real, fake):
    with tf.name_scope('d_loss'):
        real_loss = tf.reduce_mean(tf.nn.softplus(-real))
        fake_loss = tf.reduce_mean(tf.nn.softplus(fake))

        loss = real_loss + fake_loss
        tf.summary.scalar('real', real_loss)
        tf.summary.scalar('fake', fake_loss)
        tf.summary.scalar('d_entropy', loss)
    return loss


def g_loss(fakes):
    with tf.name_scope('g_loss'):
        fake, fake_ = fakes
        prob_fake = tf.reduce_mean(tf.nn.softplus(-fake_))
        not_prob_fake = tf.reduce_mean(tf.nn.softplus(fake_))

        cross_entropy = (prob_fake - not_prob_fake)
        tf.summary.scalar('fake', prob_fake)
        tf.summary.scalar('non_fake', not_prob_fake)
        tf.summary.scalar('g_entropy', cross_entropy)
    return cross_entropy


def loss(original_image, output_image, x):
    with tf.name_scope('optimizer'):

        delta = 0.0001
        sqrt = tf.abs(original_image - output_image)
        tf.summary.image('input', x)
        tf.summary.image('output', output_image)
        tf.summary.image('origin', original_image)
        cross_entropy = tf.multiply(delta, tf.reduce_sum(sqrt))
        tf.summary.scalar('entropy', cross_entropy)
    return cross_entropy


def training(loss, learning_rate, beta1, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(
            loss, global_step=global_step, var_list=var_list)

    return train_step
