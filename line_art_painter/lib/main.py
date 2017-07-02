# coding: utf-8

import math
import argparse
import numpy as np
import tensorflow as tf
from .model import model_wgan as model
import cv2

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('image', type=str, help='image to paint')
argparser.add_argument('output', type=str, help='the name of image painted')
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')

ARGS = argparser.parse_args()


def normalize_image(img):
    img = img + 1.0
    img = np.divide(img, 2.0)
    return np.multiply(img, 255.0)


def initial_image(img):
    return np.ones_like(img)


def main():
    image = cv2.imread(ARGS.image)

    width, height, chan = image.shape
    width = pow(2, math.floor(math.log2(width)))
    height = pow(2, math.floor(math.log2(height)))
    original = image[0:width, 0:height]

    image = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    image = image / 255.0
    image = np.multiply(image, 2.0)
    image = image - 1.0
    width, height = image.shape

    with tf.device('/cpu:0'):
        x = tf.placeholder(tf.float32, shape=[1, width, height])
        hint = tf.placeholder(tf.float32, shape=[1, width, height, 3])
        with tf.variable_scope('generator'):
            construction_op = model.generator(tf.reshape(x, [1, width, height, 1]), hint)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        with tf.Session() as sess:
            saver = tf.train.Saver(var_list=var_list)
            ckpt = tf.train.get_checkpoint_state(ARGS.train_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

            ret = sess.run([construction_op], {x: [image], hint: [initial_image(original)]})

            ret = normalize_image(ret[0][0])
            ret = ret.astype(np.uint8)

            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)

            cv2.imwrite(ARGS.output, ret)


if __name__ == '__main__':
    main()
