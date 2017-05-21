# coding: utf-8

import tensorflow as tf
from . import operations as op


# Define weight variable
def weight_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


# Define bias variable
def bias_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_w, patch_h, strides=[1, 1, 1, 1], name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.in_ch, self.out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(tensor, weight, strides=self.strides, padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class MaxPool(object):
    def __call__(self, conv):
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_w, patch_h, batch_size, padding='SAME', name='decoder'):
        self.batch_size = batch_size
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = padding
        self.name = name

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.out_ch, self.in_ch], name='{}_weight'.format(self.name))

        bias = bias_variable([self.out_ch], name="{}_bias".format(self.name))

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight, [self.batch_size, input_shape[0] * 2, input_shape[1] * 2, self.out_ch], [1, 2, 2, 1],
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

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd5 = op.BatchNormalization(name='bnd5')

        self.conv1 = Encoder(3, 64, 4, 4, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = Encoder(64, 128, 4, 4, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = Encoder(128, 256, 4, 4, strides=[1, 2, 2, 1], name='encoder3')
        self.conv4 = Encoder(256, 512, 4, 4, strides=[1, 2, 2, 1], name='encoder4')
        self.conv5 = Encoder(512, 1024, 4, 4, strides=[1, 2, 2, 1], name='encoder5')

        self.deconv1 = Decoder(1024, 512, 4, 4, batch_size=batch_size, name='decoder1')
        self.deconv2 = Decoder(1024, 256, 4, 4, batch_size=batch_size, name='decoder2')
        self.deconv3 = Decoder(512, 128, 4, 4, batch_size=batch_size, name='decoder3')
        self.deconv4 = Decoder(256, 64, 4, 4, batch_size=batch_size, name='decoder4')
        self.deconv5 = Decoder(128, 3, 4, 4, batch_size=batch_size, name='decoder5')


def generator(image, width, height, channels, batch_size):
    """Make construction layer.
    """

    gen = Generator(batch_size)

    relu = tf.nn.relu

    def lrelu(x):
        return tf.maximum(0.2 * x, x)

    conv1 = lrelu(gen.bnc1(gen.conv1(image, [width, height])))
    conv2 = lrelu(gen.bnc2(gen.conv2(conv1, [width // 2, height // 2])))
    conv3 = lrelu(gen.bnc3(gen.conv3(conv2, [width // 4, height // 4])))
    conv4 = lrelu(gen.bnc4(gen.conv4(conv3, [width // 8, height // 8])))
    conv5 = lrelu(gen.conv5(conv4, [width // 16, height // 16]))

    deconv1 = relu(gen.bnd1(gen.deconv1(conv5, [width // 32, height // 32])))
    deconv1 = tf.nn.dropout(deconv1, 0.5)
    deconv2 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv1, conv4], 3), [width // 16, height // 16])))
    deconv2 = tf.nn.dropout(deconv2, 0.5)
    deconv3 = relu(gen.bnd3(gen.deconv3(tf.concat([deconv2, conv3], 3), [width // 8, height // 8])))
    deconv4 = relu(gen.bnd4(gen.deconv4(tf.concat([deconv3, conv2], 3), [width // 4, height // 4])))
    deconv5 = tf.nn.tanh(gen.deconv5(tf.concat([deconv4, conv1], 3), [width // 2, height // 2]))

    return deconv5


class Discriminator(object):
    """Define discriminator"""

    def __init__(self, in_ch):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')

        self.conv1 = Encoder(in_ch, 64, 4, 4, name='encoder1', strides=[1, 2, 2, 1])
        self.conv2 = Encoder(64, 128, 4, 4, name='encoder2', strides=[1, 2, 2, 1])
        self.conv3 = Encoder(128, 256, 4, 4, name='encoder3', strides=[1, 2, 2, 1])
        self.conv4 = Encoder(256, 1, 4, 4, name='encoder4', strides=[1, 1, 1, 1])

    def __call__(self, tensor):
        _, width, height, _ = tensor.get_shape().as_list()

        def lrelu(x):
            return tf.maximum(0.2 * x, x)

        net = lrelu((self.conv1(tensor, [width, height])))
        net = lrelu(self.bnc1(self.conv2(net, [width // 2, height // 2])))
        net = lrelu(self.bnc2(self.conv3(net, [width // 4, height // 4])))
        net = lrelu((self.conv4(net, [width // 8, height // 8])))

        return net


def discriminator(bases, originals, height, width, chan):
    """make discriminator network"""

    D = Discriminator(6)
    logit = D(tf.concat([originals, bases], 3))

    tf.summary.histogram('logit', logit)
    tf.summary.histogram('softmax', tf.nn.sigmoid(logit))
    return logit


def d_loss(real, fake):
    with tf.name_scope('d_loss'):
        # log(D(x))
        real_loss = -tf.nn.softplus(-real)
        # log(1 - D(G(x)))
        fake_loss = -tf.nn.softplus(fake)

        loss = -tf.reduce_mean(real_loss + fake_loss)
        tf.summary.histogram('real', real_loss)
        tf.summary.histogram('fake', fake_loss)
        tf.summary.scalar('d_entropy', loss)
    return loss


def g_loss(logit):
    with tf.name_scope('g_loss'):
        # log(D(G(x)))
        prob_fake = -tf.nn.softplus(-logit)
        # log(1 - D(G(x)))
        not_prob_fake = -tf.nn.softplus(logit)

        # log(D(G(x)) / (1 - D(G(x))))
        # = log(D(G(x))) - log(1 - D(G(x)))
        cross_entropy = -tf.reduce_mean(prob_fake - not_prob_fake)
        tf.summary.scalar('g_entropy', cross_entropy)

    return cross_entropy


def l1_loss(original, gen):
    with tf.name_scope('l1_loss'):
        _, w, h, c = original.get_shape().as_list()
        original = tf.reshape(original, [-1, w * h * c])
        gen = tf.reshape(gen, [-1, w * h * c])

        l1_distance = 100 * tf.reduce_mean(tf.reduce_sum(tf.abs(original - gen), 1))

        tf.summary.scalar('distance', l1_distance)

    return l1_distance


class Trainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(loss, global_step=self.global_step, var_list=var_list)

        return train_step
