# coding: utf-8

import math
import tensorflow as tf
from . import model


def init_sess(batch_size, height, width, train_dir, reuse=False):

    image_size = int(math.pow(2, math.ceil(math.log2(max(height, width)))))
    x = tf.placeholder(tf.float32, [batch_size, height, width, 3])

    with tf.variable_scope('classifier', reuse=reuse):
        generate_op = model.autoencoder(x)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

    sess = tf.Session()
    saver = tf.train.Saver(var_list=var_list)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    return sess, generate_op, x


def generate(sess, op, ps, images):

    ret = sess.run(op, feed_dict={ps: images})

    return ret
