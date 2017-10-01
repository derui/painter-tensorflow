# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import pathlib
import tensorflow as tf


def cropped_image(origin, size):
    cropped = tf.random_crop(origin, size + [3])

    return cropped


def dataset_input_fn(data_dir, tfrecord, batch_size, size, distorted=True):
    file_names = [
        str(pathlib.Path(data_dir) / tfrecord)
    ]

    def read_pair(record):

        features = tf.parse_single_example(record, {
            'original': tf.FixedLenFeature([], tf.string),
        })

        original = tf.image.decode_png(features['original'], channels=3)

        reshaped_o_image = tf.cast(original, tf.float32)
        reshaped_o_image = tf.multiply(reshaped_o_image, 1 / 255.0)
        reshaped_o_image = tf.multiply(reshaped_o_image, 2) - 1.0
        reshaped_o_image = cropped_image(reshaped_o_image, [size, size])

        return {'original': reshaped_o_image}

    dataset = tf.contrib.data.TFRecordDataset(file_names)
    dataset = dataset.map(read_pair)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()['original']
