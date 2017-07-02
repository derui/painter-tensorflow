# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import os
import random
import tensorflow as tf
import sys


def read_pair(filename_queue):
    class Record(object):
        pass

    result = Record()

    reader = tf.TFRecordReader()

    result.key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, {
        'painted': tf.FixedLenFeature([], tf.string),
        'line_art': tf.FixedLenFeature([], tf.string),
    })

    painted = tf.image.decode_png(features['painted'], channels=3)
    line_art = tf.image.decode_png(features['line_art'], channels=1)

    result.painted = painted
    result.line_art = line_art

    return result


def cropped_image(origin, line, size):
    concat = tf.concat([origin, line], 2)
    cropped = tf.random_crop(concat, size + [4])

    return (cropped[:, :, :3], cropped[:, :, 3:])


def distorted_image(origin, wire):
    """Construct distorted image for training"""
    seed = random.random()
    lr_random_seed = tf.random_uniform([], seed=seed)
    lr_where = tf.greater_equal(lr_random_seed, 0.5)
    origin = tf.where(lr_where, origin, tf.image.flip_left_right(origin))
    wire = tf.where(lr_where, wire, tf.image.flip_left_right(wire))

    ud_random_seed = tf.random_uniform([], seed=random.random())
    ud_where = tf.greater_equal(ud_random_seed, 0.5)
    origin = tf.where(ud_where, origin, tf.image.flip_up_down(origin))
    wire = tf.where(ud_where, wire, tf.image.flip_up_down(wire))

    origin = tf.image.random_hue(origin, max_delta=0.5)
    # origin = tf.image.random_contrast(origin, 0.5, 1.0)
    return origin, wire


def _generate_pair_batch(pair, min_queue_examples, batch_size, shuffle):
    """
    Generate image pair batch. Return images of pair as (original, base).
    """

    num_preprocess_threads = 3

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


def inputs(directory, batch_size, size, distorted=True):

    file_names = []
    for (root, _, files) in os.walk(directory):
        for f in files:
            file_names.append(os.path.join(root, f))

    filename_queue = tf.train.string_input_producer(file_names)
    num_examples_per_epoch = 30

    read_input = read_pair(filename_queue)
    reshaped_o_image, reshaped_w_image = cropped_image(
        read_input.painted, read_input.line_art, [size, size])
    if distorted:
        reshaped_o_image, reshaped_w_image = distorted_image(reshaped_o_image,
                                                             reshaped_w_image)

    reshaped_o_image = tf.cast(reshaped_o_image, tf.float32)
    reshaped_o_image = tf.multiply(reshaped_o_image, 1 / 255.0)
    reshaped_w_image = tf.cast(reshaped_w_image, tf.float32)
    reshaped_w_image = tf.multiply(reshaped_w_image, 1 / 255.0)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_pair_batch(
        [reshaped_o_image, reshaped_w_image],
        min_queue_examples,
        batch_size,
        shuffle=True)
