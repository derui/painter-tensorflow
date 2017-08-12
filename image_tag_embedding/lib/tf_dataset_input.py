# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import numpy as np
import os
import pathlib
import tensorflow as tf


def read_tags(filename_queue, max_document_length):
    class Record(object):
        pass

    result = Record()

    reader = tf.TFRecordReader()

    result.key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, {
        'tags': tf.FixedLenFeature([max_document_length], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
    })

    result.image = tf.reshape(
        tf.image.decode_png(features['image'], 3), [128, 128, 3])
    result.tags = features['tags']

    return result


def _generate_batch(data, min_queue_examples, batch_size, shuffle):
    """
    Generate image pair batch. Return images of pair as (original, base).
    """

    num_preprocess_threads = 1

    if shuffle:
        tags = tf.train.shuffle_batch(
            data,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        tags = tf.train.batch(
            data,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return tags


def inputs(data_dir, batch_size, max_document_length):
    file_names = []
    for (root, _, files) in os.walk(data_dir):
        for f in files:
            file_names.append(str(pathlib.Path(root) / f))

    filename_queue = tf.train.string_input_producer(file_names)
    num_examples_per_epoch = 100

    read_input = read_tags(filename_queue, max_document_length)
    tags = read_input.tags
    image = tf.cast(read_input.image, tf.float32)
    image = tf.multiply(image, 1 / 255.0)
    image = tf.multiply(image, 2) - 1.0

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_batch([tags, image],
                           min_queue_examples, batch_size, shuffle=True)
