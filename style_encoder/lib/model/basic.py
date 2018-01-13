# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class Encoder(object):
    def __init__(self, channels):

        self.bnc0 = op.BatchNormalization(name='bnc0')
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')

        self.conv0 = op.Encoder(channels, 32, 3, 3, strides=[1, 1, 1, 1], name='encoder0')
        self.conv1 = op.Encoder(32, 64, 4, 4, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = op.Encoder(64, 128, 4, 4, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = op.Encoder(128, 256, 4, 4, strides=[1, 2, 2, 1], name='encoder3')
        self.conv4 = op.Encoder(256, 512, 4, 4, strides=[1, 2, 2, 1], name='encoder4')

        self.dense = op.Dense()


def encode(image):

    channels = image.shape.as_list()[3]
    gen = Encoder(channels)

    relu = tf.nn.relu

    conv0 = relu(gen.bnc0(gen.conv0(image)))
    conv1 = relu(gen.bnc1(gen.conv1(conv0)))
    conv2 = relu(gen.bnc2(gen.conv2(conv1)))
    conv3 = relu(gen.bnc3(gen.conv3(conv2)))
    conv4 = relu(gen.bnc4(gen.conv4(conv3)))

    dense = gen.dense(conv4, 1024)

    return conv4, dense


class Decoder(object):
    def __init__(self):
        self.bnd1 = op.BatchNormalization(name='bnd2')
        self.bnd2 = op.BatchNormalization(name='bnd3')
        self.bnd3 = op.BatchNormalization(name='bnd4')
        self.bnd4 = op.BatchNormalization(name='bnd5')

        self.deconv4 = op.PixelShuffler(op.Encoder(512, 1024, 3, 3, name='decoder4'), 256, 2)
        self.deconv3 = op.PixelShuffler(op.Encoder(256, 512, 3, 3, name='decoder3'), 128, 2)
        self.deconv2 = op.PixelShuffler(op.Encoder(256, 256, 3, 3, name='decoder2'), 64, 2)
        self.deconv1 = op.PixelShuffler(op.Encoder(128, 128, 3, 3, name='decoder1'), 32, 2)
        self.deconv0 = op.Encoder(32, 3, 3, 3, name="decoder0")

        self.dense = op.Dense('dense')


def decode(fcl, pre_dense):
    """Make construction layer.
    """

    D = Decoder()

    relu = tf.nn.relu

    _, c = fcl.get_shape().as_list()
    _, h, w, c = pre_dense.get_shape().as_list()
    net = D.dense(fcl, h * w * c)
    net = tf.reshape(net, [-1, h, w, c])

    net = relu(D.bnd4(D.deconv4(net)))
    net = relu(D.bnd3(D.deconv2(net)))
    net = relu(D.bnd2(D.deconv3(net)))
    net = relu(D.bnd1(D.deconv1(net)))
    net = tf.nn.tanh(D.deconv0(net))

    return net


def loss(gen, real):

    loss = tf.reduce_mean(tf.abs(real - gen), [1,2,3])

    loss = tf.reduce_mean(loss)
    return loss


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
