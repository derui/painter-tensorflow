# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import os
import tensorflow as tf


def read_pair(filename_queue):
    class Record(object):
        pass

    result = Record()

    reader = tf.TFRecordReader()

    result.key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, {
        'original': tf.FixedLenFeature([], tf.string),
    })

    original = tf.image.decode_png(features['original'], channels=3)

    result.original = original
    return result


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


def cropped_image(origin, size):
    cropped = tf.random_crop(origin, size + [3])

    return cropped


def _generate_pair_batch(pair, min_queue_examples, batch_size, shuffle):
    """
    Generate image pair batch. Return images of pair as (original, base).
    """

    num_preprocess_threads = 1

    if shuffle:
        images = tf.train.shuffle_batch(
            pair,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        images = tf.train.batch(
            pair,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images


def inputs(data_dir, batch_size, size, distorted=True):
    file_names = []
    for (root, _, files) in os.walk(data_dir):
        for f in files:
            file_names.append(os.path.join(root, f))

    filename_queue = tf.train.string_input_producer(file_names)
    num_examples_per_epoch = 100

    read_input = read_pair(filename_queue)
    reshaped_o_image = tf.cast(read_input.original, tf.float32)
    reshaped_o_image = tf.multiply(reshaped_o_image, 1 / 255.0)
    reshaped_o_image = tf.multiply(reshaped_o_image, 2) - 1.0
    reshaped_o_image = cropped_image(reshaped_o_image, [size, size])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_pair_batch([reshaped_o_image], min_queue_examples, batch_size, shuffle=True)
