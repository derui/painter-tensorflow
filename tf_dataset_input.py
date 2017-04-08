# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import os
import tensorflow as tf
from tools import dataset as ds


def read_pair(filename_queue):
    class Record(object):
        pass

    result = Record()
    record_bytes = ds.RECORD_SIZE

    reader = tf.FixedLengthRecordReader(record_bytes)

    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.original_image = tf.reshape(record_bytes[0:ds.IMAGE_SIZE],
                                       ds.original_shape())
    result.wire_frame_image = tf.reshape(record_bytes[ds.IMAGE_SIZE:],
                                         ds.line_art_shape())

    return result


def distorted_image(origin, wire):
    """Construct distorted image for training"""

    lr_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(lr_random, .5)

    origin = tf.where(mirror_cond, origin, tf.image.flip_left_right(origin))
    wire = tf.where(mirror_cond, wire, tf.image.flip_left_right(wire))

    return origin, wire


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


def inputs(data_dir, batch_size, distorted=True):
    filename_range = 0
    for (root, _, files) in os.walk(data_dir):
        filename_range += len(files)

    filenames = [
        os.path.join(data_dir, 'image_pack_{}.bin'.format(i + 1))
        for i in range(0, filename_range)
    ]

    filename_queue = tf.train.string_input_producer(filenames)
    num_examples_per_epoch = 100

    read_input = read_pair(filename_queue)
    reshaped_o_image = tf.cast(read_input.original_image, tf.float32)
    reshaped_o_image = tf.multiply(reshaped_o_image, 1 / 255.0)
    reshaped_o_image = tf.multiply(reshaped_o_image, 2) - 1.0
    reshaped_w_image = tf.cast(read_input.wire_frame_image, tf.float32)
    reshaped_w_image = tf.multiply(reshaped_w_image, 1 / 255.0)
    reshaped_w_image = tf.multiply(reshaped_w_image, 2) - 1.0

    if distorted:
        reshaped_o_image, reshaped_w_image = distorted_image(reshaped_o_image,
                                                             reshaped_w_image)
    tf.train.MonitoredTrainingSession

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_pair_batch(
        [reshaped_o_image, reshaped_w_image],
        min_queue_examples,
        batch_size,
        shuffle=True)
