# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class EmbeddingEncoder(object):
    """Define encoder for embedding"""

    def __init__(self, sequence_length, embedding_size,
                 filter_sizes=[2, 3, 4]):
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes

        def gen_encoder(index, filter_size, embedding_size, sequence_length):

            encoder = op.Encoder(
                1,
                128,
                filter_size,
                embedding_size,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='encoder/{}'.format(index))

            max_pool = op.MaxPool(
                [1, sequence_length - filter_size + 1, 1, 1],
                name="max_pooling/{}".format(index))

            def encode(tensor):
                return max_pool(tf.nn.relu(encoder(tensor)))

            return encode

        self.encoders = [
            gen_encoder(i, filter_sizes[i], embedding_size, sequence_length)
            for i in range(len(filter_sizes))
        ]

        self.linear_encoder = op.LinearEncoder(sequence_length)

    def __call__(self, tensor):

        encoded = [encoder(tensor) for encoder in self.encoders]

        total_channels = 128 * len(self.filter_sizes)
        net = tf.concat(encoded, 3)
        net = tf.reshape(net, [-1, total_channels])

        return self.linear_encoder(net, total_channels)


class EmbeddingDecoder(object):
    """Define encoder for image as description"""

    def __init__(self):
        self.linear_encoder = op.LinearEncoder(16 * 16 * 256)
        self.dec4 = op.PixelShuffler(op.Encoder(256, 1024, 3, 3, name='decoder/4'), 256, 2)
        self.dec3 = op.PixelShuffler(op.Encoder(256, 512, 3, 3, name='decoder/3'), 128, 2)
        self.dec2 = op.PixelShuffler(op.Encoder(128, 256, 3, 3, name='decoder/2'), 64, 2)
        self.dec1 = op.Encoder(64, 3, 3, 3, name='decoder/1')
        
        self.bnd1 = op.BatchNormalization(name='bnd/1')
        self.bnd2 = op.BatchNormalization(name='bnd/2')
        self.bnd3 = op.BatchNormalization(name='bnd/3')

    def __call__(self, tensor, input_channel):

        relu = tf.nn.relu
        net = self.linear_encoder(tensor, input_channel)
        net = tf.reshape(net, [-1, 16, 16, 256])
        net = self.bnd3(relu(self.dec4(net)))
        net = self.bnd2(relu(self.dec3(net)))
        net = self.bnd1(relu(self.dec2(net)))
        net = tf.nn.tanh(self.dec1(net))

        return net


def embedding_encoder(tag):
    """make embedding encoder network"""
    shape = tag.shape.as_list()
    with tf.variable_scope('embedding_encoder'):
        net = EmbeddingEncoder(shape[1], shape[2])(tag)

    return net


def embedding_decoder(logit):
    """make image encoder network"""
    with tf.variable_scope('embedding_decoder'):
        net = EmbeddingDecoder()(logit, logit.shape.as_list()[1])

    return net


def loss(decoded, original):

    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(decoded - original), 1))

    return loss


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(
            loss, global_step=global_step, var_list=var_list)

    return train_step
