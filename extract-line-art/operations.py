# -*- coding: utf-8 -*-
import tensorflow as tf


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
                x,
                self.gamma,
                self.beta,
                epsilon=self.epsilon)

        return y
