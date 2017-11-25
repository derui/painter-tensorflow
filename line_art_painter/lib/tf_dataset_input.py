# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import os
import tensorflow as tf


def distorted_image(origin, wire):
    """Construct distorted image for training"""

    lr_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(lr_random, .5)

    origin = tf.where(mirror_cond, origin, tf.image.flip_left_right(origin))
    wire = tf.where(mirror_cond, wire, tf.image.flip_left_right(wire))

    ud_random = tf.random_uniform([], 0, 1.0)
    ud_cond = tf.less(ud_random, .5)

    origin = tf.where(ud_cond, origin, tf.image.flip_up_down(origin))
    wire = tf.where(ud_cond, wire, tf.image.flip_up_down(wire))

    return origin, wire


def dataset_input_fn(data_dir, batch_size, distorted=True):
    file_names = []
    for (root, _, files) in os.walk(data_dir):
        for f in files:
            file_names.append(os.path.join(root, f))

    def read_pair(record):

        features = tf.parse_single_example(record, {
            'original': tf.FixedLenFeature([], tf.string),
            'line_art': tf.FixedLenFeature([], tf.string),
        })

        original = tf.image.decode_png(features['original'], channels=3)
        line_art = tf.image.decode_png(features['line_art'], channels=1)

        original = tf.reshape(original, [128, 128, 3])
        line_art = tf.reshape(line_art, [128, 128, 1])

        if distorted:
            original, line_art = distorted_image(original, line_art)
        original = tf.cast(original, tf.float32)
        original = tf.multiply(original, 1 / 255.0)
        original = tf.multiply(original, 2) - 1.0
        line_art = tf.cast(line_art, tf.float32)
        line_art = tf.multiply(line_art, 1 / 255.0)
        line_art = tf.multiply(line_art, 2) - 1.0

        return {'original': original, 'line_art': line_art}

    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(read_pair, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=100*batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()

    next_example = iterator.get_next()
    return iterator, next_example['original'], next_example['line_art']
