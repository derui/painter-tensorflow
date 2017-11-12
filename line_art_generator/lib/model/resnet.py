# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class AutoEncoder(object):
    """Define autoencoder"""

    def __init__(self):
        self.conv1 = op.Encoder(3, 48, 5, 5, strides=[1, 2, 2, 1], name='encoder1')
        self.conv1_f1 = op.Encoder(48, 128, 3, 3, name='encoder1_flat1')
        self.conv1_f2 = op.Encoder(128, 128, 3, 3, name='encoder1_flat2')
        self.conv2 = op.Encoder(128, 256, 5, 5, strides=[1, 2, 2, 1], name='encoder2')
        self.conv2_f1 = op.Encoder(256, 256, 3, 3, name='encoder2_flat1')
        self.conv2_f2 = op.Encoder(256, 256, 3, 3, name='encoder2_flat2')
        self.conv3 = op.Encoder(256, 256, 5, 5, strides=[1, 2, 2, 1], name='encoder3')
        self.conv3_f1 = op.Encoder(256, 512, 3, 3, name='encoder3_flat1')
        self.conv3_f2 = op.Encoder(512, 1024, 3, 3, name='encoder3_flat2')
        self.conv3_f3 = op.Encoder(1024, 1024, 3, 3, name='encoder3_flat3')

        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc1_f1 = op.BatchNormalization(name='bnc1_flat1')
        self.bnc1_f2 = op.BatchNormalization(name='bnc1_flat2')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc2_f1 = op.BatchNormalization(name='bnc2_flat1')
        self.bnc2_f2 = op.BatchNormalization(name='bnc2_flat2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc3_f1 = op.BatchNormalization(name='bnc3_flat1')
        self.bnc3_f2 = op.BatchNormalization(name='bnc3_flat2')
        self.bnc3_f3 = op.BatchNormalization(name='bnc3_flat3')

        self.deconv3 = op.PixelShuffler(None, 256, 2)
        self.deconv3_f1 = op.Encoder(256, 256, 3, 3, name='decoder3_flat1')
        self.deconv3_f2 = op.Encoder(256, 256, 3, 3, name='decoder3_flat2')
        self.deconv2 = op.PixelShuffler(op.Encoder(256, 512, 5, 5, name='decoder2'), 128, 2)
        self.deconv2_f1 = op.Encoder(128, 128, 3, 3, name='decoder2_flat1')
        self.deconv2_f2 = op.Encoder(128, 128, 3, 3, name='decoder2_flat2')
        self.deconv1 = op.PixelShuffler(op.Encoder(128, 256, 5, 5, name='decoder1'), 64, 2)
        self.deconv1_f1 = op.Encoder(64, 32, 3, 3, name='decoder1_flat1')
        self.deconv1_f2 = op.Encoder(32, 32, 3, 3, name='decoder1_flat2')
        self.deconv0 = op.Encoder(32, 1, 3, 3, name='decoder0')

        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd3_f1 = op.BatchNormalization(name='bnd3_flat1')
        self.bnd3_f2 = op.BatchNormalization(name='bnd3_flat2')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd2_f1 = op.BatchNormalization(name='bnd2_flat1')
        self.bnd2_f2 = op.BatchNormalization(name='bnd2_flat2')
        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd1_f1 = op.BatchNormalization(name='bnd1_flat1')
        self.bnd1_f2 = op.BatchNormalization(name='bnd1_flat2')


def autoencoder(images, scale=1.0):
    """make autoencoder network"""

    AE = AutoEncoder()

    relu = tf.nn.relu
    net = relu(AE.bnc1(AE.conv1(images)))
    net = relu(AE.bnc1_f1(AE.conv1_f1(net)))
    net = relu(AE.bnc1_f2(AE.conv1_f2(net)))
    net = relu(AE.bnc2(AE.conv2(net)))
    net = relu(AE.bnc2_f1(AE.conv2_f1(net)))
    net = relu(AE.bnc2_f2(AE.conv2_f2(net)))
    net = relu(AE.bnc3(AE.conv3(net)))
    net = relu(AE.bnc3_f1(AE.conv3_f1(net)))
    net = relu(AE.bnc3_f2(AE.conv3_f2(net)))
    net = relu(AE.bnc3_f3(AE.conv3_f3(net)))
    net = relu(AE.bnd3(AE.deconv3(net)))
    net = relu(AE.bnd3_f1(AE.deconv3_f1(net)))
    net = relu(AE.bnd3_f2(AE.deconv3_f2(net)))
    net = relu(AE.bnd2(AE.deconv2(net)))
    net = relu(AE.bnd2_f1(AE.deconv2_f1(net)))
    net = relu(AE.bnd2_f2(AE.deconv2_f2(net)))
    net = relu(AE.bnd1(AE.deconv1(net)))
    net = relu(AE.bnd1_f1(AE.deconv1_f1(net)))
    net = relu(AE.bnd1_f2(AE.deconv1_f2(net)))

    net = tf.nn.sigmoid(AE.deconv0(net))

    return net


def loss_map(target, bins, alpha, beta):
    """construct loss map from target image"""
    with tf.name_scope('loss_map'):

        patches = tf.extract_image_patches(target, [1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
        minimum = tf.reduce_min(patches, axis=3, keep_dims=True)
        maximum = tf.reduce_max(patches, axis=3, keep_dims=True)

        normalized = (patches - minimum) / (maximum - minimum)
        normalized = tf.floor(patches * bins) / bins

        # make histogram and suppress illegal histogram
        hist = tf.reshape(normalized[:, :, :, 12], [-1] + target.shape.as_list()[1:])
        hist = tf.where(tf.equal(minimum, maximum), minimum, hist)

        ones = tf.ones_like(target)
        eq = tf.equal(1.0, target)
        hist = tf.where(eq, ones, tf.minimum(1.0, alpha * tf.exp(-hist) + beta))

    return hist


def loss(target, gen, lossmap):

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(lossmap * (gen - target))))

        tf.summary.image('loss_applied', tf.square(lossmap * (gen - target)))

        tf.summary.scalar('loss', loss)
    return loss


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_step
