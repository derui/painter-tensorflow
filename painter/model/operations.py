# -*- coding: utf-8 -*-
import tensorflow as tf
import math


class LinearEncoder(object):
    """Encoder for Linear Operation."""

    def __init__(self, name='linear_encoder'):
        self.name = name

    def __call__(self, tensor, in_ch, out_ch):
        weight = tf.get_variable(
            "{}_weight".format(self.name)[in_ch, out_ch], initializer=tf.random_uniform_initializer(stddev=0.02))
        bias = tf.get_variable('{}_bias'.format(self.name), [out_ch], initializer=tf.constant_initializer(0.0))
        conv = tf.matmul(tensor, weight)
        conv = tf.nn.bias_add(conv, bias)

        return conv


class BatchNormalization(object):
    def __init__(self, epsilon=0.0005, momentam=0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentam = momentam
        self.name = name

    def __call__(self, x, train=True):
        """Implementation batch_normalization """

        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1.0, 0.1))

            y, _, _ = tf.nn.fused_batch_norm(x, self.gamma, self.beta, epsilon=self.epsilon)

        return y


class PixelShuffler(object):
    """Declare PixelShuffler.
    """

    def __init__(self, conv, out_ch, scale):
        """
        @param conv convolution layer such as encoder. This layer should be able to
                    convert input channels to output channels ** scale.
        @param out_ch the channels of output want to convert
        @param scale scaling of PixelShuffler
        """
        self.conv = conv
        self.scale = scale
        self.out_ch = out_ch

    def __call__(self, tensor):
        """
        @param tensor the tensor to upscale via PixelShuffler
        """

        net = tensor
        if self.conv is not None:
            net = self.conv(tensor)

        batch_size, h, w, ic = tensor.shape.as_list()
        r = self.scale
        f_h, f_w = math.floor(h * r), math.floor(w * r)
        net = tf.reshape(net, [batch_size, w, h, r, r, self.out_ch])
        net = tf.transpose(net, [0, 1, 3, 2, 4, 5])
        net = tf.reshape(net, [batch_size, f_h, f_w, self.out_ch])

        return net
