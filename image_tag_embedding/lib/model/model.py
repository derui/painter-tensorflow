# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class EmbeddingEncoder(object):
    """Define encoder for embedding"""

    def __init__(self, sequence_length, embedding_size, trainable=True, filter_sizes=[2, 3, 4]):
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
                trainable=trainable,
                name='encoder/{}'.format(index))

            max_pool = op.MaxPool([1, sequence_length - filter_size + 1, 1, 1], name="max_pooling/{}".format(index))

            def encode(tensor):
                return max_pool(tf.nn.relu(encoder(tensor)))

            return encode

        self.encoders = [
            gen_encoder(i, filter_sizes[i], embedding_size, sequence_length) for i in range(len(filter_sizes))
        ]

        self.linear_encoder = op.LinearEncoder(sequence_length, trainable=trainable)

    def __call__(self, tensor):

        encoded = [encoder(tensor) for encoder in self.encoders]

        total_channels = 128 * len(self.filter_sizes)
        net = tf.concat(encoded, 3)
        net = tf.reshape(net, [-1, total_channels])

        return self.linear_encoder(net, total_channels)


class ImageAutoEncoder(object):
    """Define autoencoder for image as description"""

    def __init__(self, embedding_channels):
        self.enc1 = op.Encoder(3, 64, 3, 3, strides=[1, 2, 2, 1], name='encoder/1')
        self.enc2 = op.Encoder(64, 128, 3, 3, strides=[1, 2, 2, 1], name='encoder/2')
        self.enc3 = op.Encoder(128, 256, 3, 3, strides=[1, 2, 2, 1], name='encoder/3')
        self.enc4 = op.Encoder(256, 512, 3, 3, strides=[1, 1, 1, 1], name='encoder/4')

        self.dec4 = op.PixelShuffler(op.Encoder(512 + embedding_channels, 1024, 3, 3, name='decoder/4'), 256, 2)
        self.dec3 = op.PixelShuffler(op.Encoder(256, 512, 3, 3, name='decoder/3'), 128, 2)
        self.dec2 = op.PixelShuffler(op.Encoder(128, 256, 3, 3, name='decoder/2'), 64, 2)
        self.dec1 = op.Encoder(64, 3, 3, 3, name='decoder/1')

        self.bnd1 = op.BatchNormalization(name='bnd/1')
        self.bnd2 = op.BatchNormalization(name='bnd/2')
        self.bnd3 = op.BatchNormalization(name='bnd/3')

        self.bnc1 = op.BatchNormalization(name='bnc/1')
        self.bnc2 = op.BatchNormalization(name='bnc/2')
        self.bnc3 = op.BatchNormalization(name='bnc/3')
        self.bnc4 = op.BatchNormalization(name='bnc/4')

    def __call__(self, image, embedding):

        relu = tf.nn.relu
        net = self.bnc1(relu(self.enc1(image)))
        net = self.bnc2(relu(self.enc2(net)))
        net = self.bnc3(relu(self.enc3(net)))
        net = self.bnc4(relu(self.enc4(net)))
        shape = net.shape.as_list()
        original_size = embedding.shape.as_list()[1]
        embedding = tf.tile(embedding, [1, shape[1] * shape[2]])
        embedding = tf.reshape(embedding, [-1, shape[1], shape[2], original_size])

        net = self.bnd3(relu(self.dec4(tf.concat([net, embedding], axis=3))))
        net = self.bnd2(relu(self.dec3(net)))
        net = self.bnd1(relu(self.dec2(net)))
        net = tf.nn.tanh(self.dec1(net))

        return net


def embedding_encoder(tag, trainable=True):
    """make embedding encoder network"""
    shape = tag.shape.as_list()
    with tf.variable_scope('embedding_encoder'):
        net = EmbeddingEncoder(shape[1], shape[2], trainable=trainable)(tag)

    return net


def image_autoencoder(image, embedding):
    """make image autoencoder"""
    with tf.variable_scope('image_autoencoder'):
        shape = embedding.shape.as_list()
        net = ImageAutoEncoder(shape[1])(image, embedding)

    return net


def loss(decoded, original):

    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(decoded - original), 1))

    return loss


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_step
