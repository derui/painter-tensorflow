# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class Upsampler(object):
    def __init__(self, channels):
        self.bnc1 = op.BatchNormalization(name='upsampler/bnc1')
        self.bnc2 = op.BatchNormalization(name='upsampler/bnc2')
        self.bnc3 = op.BatchNormalization(name='upsampler/bnc3')
        self.bnc4 = op.BatchNormalization(name='upsampler/bnc4')

        self.conv1 = op.Encoder(channels, 256, 3, 3, name='upsampler/encoder0')

        self.conv2 = op.PixelShuffler(op.Encoder(256, 1024, 3, 3, name='upsampler/encoder1'), 256, 2)
        self.resnet1 = op.ResNet(256, name='upsampler/resnet1')
        self.conv3 = op.Encoder(256, 128, 3, 3, name='upsampler/encoder3')
        self.resnet2 = op.PixelShuffler(op.ResNet(128, name='upsampler/resnet2'), 32, 2)
        self.resnet3 = op.ResNet(32, name='upsampler/resnet3')
        self.conv4 = op.Encoder(32, 3, 3, 3, name='upsampler/encoder4')

    def __call__(self, tensor):

        net = tensor
        net = tf.nn.relu(self.bnc1(self.conv1(net)))
        net = tf.nn.relu(self.bnc2(self.conv2(net)))
        net = self.resnet1(net)
        net = tf.nn.relu(self.bnc3(self.conv3(net)))
        net = self.resnet2(net)
        net = self.resnet3(net)

        return tf.nn.tanh(self.conv4(net))


def upsampler(small):
    """Make construction layer.
    """
    channels = small.shape.as_list()[3]
    upsampler = Upsampler(channels)

    return upsampler(small)


def l1_loss(original, gen):
    l1_distance = tf.reduce_mean(tf.abs(original - gen)) * 100

    return l1_distance


class AdamTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list=None):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(
            loss, global_step=self.global_step, var_list=var_list, colocate_gradients_with_ops=True)

        return train_step
