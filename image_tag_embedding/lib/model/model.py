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


class ImageEncoder(object):
    """Define encoder for image as description"""

    def __init__(self, sequence_length):
        self.enc1 = op.Encoder(3, 64, 4, 4, strides=[1,2,2,1], name="encoder/1")
        self.enc2 = op.Encoder(64, 128, 4, 4, strides=[1,2,2,1], name="encoder/2")
        self.enc3 = op.Encoder(128, 256, 4, 4, strides=[1,2,2,1], name="encoder/3")
        
        self.bnc1 = op.BatchNormalization(name="bnc/1")
        self.bnc2 = op.BatchNormalization(name="bnc/2")
        self.bnc3 = op.BatchNormalization(name="bnc/3")

        self.linear_encoder = op.LinearEncoder(sequence_length)

    def __call__(self, tensor):

        relu = tf.nn.relu
        net = self.bnc1(relu(self.enc1(tensor)))
        net = self.bnc2(relu(self.enc2(net)))
        net = self.bnc3(relu(self.enc3(net)))

        shape = net.shape.as_list()
        net = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])

        return self.linear_encoder(net, shape[1] * shape[2] * shape[3])


def embedding_encoder(tag):
    """make embedding encoder network"""
    shape = tag.shape.as_list()
    with tf.variable_scope('embedding_encoder'):
        net = EmbeddingEncoder(shape[1], shape[2])(tag)

    return net


def image_encoder(image, sequence_length):
    """make image encoder network"""
    with tf.variable_scope('image_encoder'):
        net = ImageEncoder(sequence_length)(image)

    return net


def loss(emb, image, tag):
    tag = tf.cast(tag, tf.float32)
    e_loss = tf.nn.softmax_cross_entropy_with_logits(logits=emb, labels=tag)
    i_loss = tf.nn.softmax_cross_entropy_with_logits(logits=image, labels=tag)

    return tf.reduce_mean(e_loss + i_loss)


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(
            loss, global_step=global_step, var_list=var_list)

    return train_step
