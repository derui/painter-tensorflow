# -*- coding: utf-8 -*-
"""
Tensorflow specialized image reader.

This module optimized to use filename_queue and batch
without overhead coping data.
"""

import pathlib
import random
import tensorflow as tf


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


def dataset_input_fn(directory, batch_size, size, distorted=True):
    def read_pair(record):
        features = tf.parse_single_example(record, {
            'painted': tf.FixedLenFeature([], tf.string),
            'line_art': tf.FixedLenFeature([], tf.string),
        })

        painted = tf.image.decode_png(features['painted'], channels=3)
        line_art = tf.image.decode_png(features['line_art'], channels=1)

        painted, line_art = cropped_image(painted, line_art, [size, size])
        if distorted:
            painted, line_art = distorted_image(painted, line_art)

        painted = tf.cast(painted, tf.float32)
        painted = tf.multiply(painted, 1 / 255.0)
        line_art = tf.cast(line_art, tf.float32)
        line_art = tf.multiply(line_art, 1 / 255.0)

        return painted, line_art

    file_names = [str(pathlib.Path(directory) / "out.tfrecords")]

    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(read_pair, num_parallel_calls=8).prefetch(batch_size)
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    next_data = iterator.get_next()
    return iterator, next_data
