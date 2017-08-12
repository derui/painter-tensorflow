# -*- coding: utf-8 -*-
import tensorflow as tf
import math


# Define weight variable
def weight_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


# Define bias variable
def bias_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0.0))


class MaxPool(object):
    """Max pooling"""

    def __init__(self,
                 ksize,
                 strides=[1, 1, 1, 1],
                 name='max_pool'):
        self.name = name
        self.ksize = ksize
        self.strides = strides

    def __call__(self, tensor):
        """Return tensor applied max-pooling"""
        return tf.nn.max_pool(tensor,
                              self.ksize,
                              self.strides,
                              "VALID",
                              name=self.name)


class LinearEncoder(object):
    """Encoder for Linear Operation."""

    def __init__(self, out_ch, name='linear_encoder'):
        self.name = name
        self.out_ch = out_ch

    def __call__(self, tensor, in_ch):
        weight = tf.get_variable(
            "{}_weight".format(self.name), [in_ch, self.out_ch],
            initializer=tf.random_uniform_initializer())
        bias = tf.get_variable(
            '{}_bias'.format(self.name), [self.out_ch],
            initializer=tf.constant_initializer(0.0))
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
            self.beta = tf.get_variable(
                "beta", [shape[-1]], initializer=tf.constant_initializer(0.))
            self.gamma = tf.get_variable(
                "gamma", [shape[-1]],
                initializer=tf.random_normal_initializer(1.0, 0.1))

            y, _, _ = tf.nn.fused_batch_norm(
                x, self.gamma, self.beta, epsilon=self.epsilon)

        return y


class LayerNormalization(object):
    def __init__(self, epsilon=0.0005, name="layer_norm"):
        self.epsilon = epsilon
        self.name = name

    def __call__(self, x, train=True):
        """Implementation layer_normalization """
        # keep only neurons on [BHWC] format.
        neurons = x.shape.as_list()[3]
        with tf.variable_scope(self.name) as scope:

            # compute mean and variance of each neuron.
            # 1/HΣi=>H ai , resulting only one value each input.
            mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

            self.beta = tf.get_variable("beta", [neurons],
                                        initializer=tf.constant_initializer(0.))
            self.gamma = tf.get_variable("gamma", [neurons],
                                         initializer=tf.constant_initializer(1.))
            # broadcasting dims for [BHWC]
            self.beta = tf.reshape(self.beta, [1, 1, -1])
            self.gamma = tf.reshape(self.gamma, [1, 1, -1])

            y = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)

        return y


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 patch_h,
                 patch_w,
                 strides=[1, 1, 1, 1],
                 padding="SAME",
                 name='encoder'):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.padding = padding
        self.strides = strides

    def __call__(self, tensor):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.in_ch, self.out_ch],
            name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(
            tensor, weight, strides=self.strides, padding=self.padding)
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 patch_h,
                 patch_w,
                 batch_size,
                 strides=[1, 1, 1, 1],
                 name='decoder'):
        self.batch_size = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, output_shape):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.out_ch, self.in_ch],
            name='{}_weight'.format(self.name))

        bias = bias_variable([self.out_ch], name="{}_bias".format(self.name))

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight,
            [self.batch_size, output_shape[0], output_shape[1], self.out_ch],
            strides=self.strides,
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


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


class Dense(object):
    """The dense layer """

    def __init__(self, name='dense'):
        self.name = name

    def __call__(self, tensor, out_ch):
        _, w, h, c = tensor.shape.as_list()
        in_ch = w * h * c
        weight = weight_variable(
            [in_ch, out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.bias_add(
            tf.matmul(tf.reshape(tensor, [-1, in_ch]), weight), bias)

        return conv
