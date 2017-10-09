# coding: utf-8

import math
import argparse
import numpy as np
import tensorflow as tf
from dnn_upscaler.lib.model import model as upscaler
import cv2

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('image', type=str, help='image to paint')
argparser.add_argument('output', type=str, help='the name of image painted')
argparser.add_argument('--model', type=str, help='model type')
argparser.add_argument(
    '--painter_checkpoint_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument(
    '--scaler_checkpoint_dir', default='./log', type=str, help='Directory will have been saving checkpoint')

ARGS = argparser.parse_args()

NOISE_SIZE = 128


def normalize_image(img):
    img = img + 1.0
    img = np.divide(img, 2.0)
    return np.multiply(img, 255.0)


def main():

    if ARGS.model == "wgan":
        from .lib.model import model_wgan as model
    elif ARGS.model == "began":
        from .lib.model import model_sd_began as model

    image = cv2.imread(ARGS.image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255.0
    image = np.multiply(image, 2.0)
    image = image - 1.0
    height, width = image.shape
    s_height, s_width = (height // 4, width // 4)

    work_h = int(math.pow(2, math.ceil(math.log2(s_height))))
    work_w = int(math.pow(2, math.ceil(math.log2(s_width))))

    noise = tf.random_uniform([1, NOISE_SIZE], minval=-1.0, maxval=1.0, dtype=tf.float32)

    original_size = tf.placeholder(tf.float32, shape=[1, height, width, 1])
    x = tf.placeholder(tf.float32, shape=[1, height, width])
    x_ = tf.reshape(x, [1, height, width, 1])
    x_ = tf.image.resize_images(x_, (s_height, s_width), method=tf.image.ResizeMethod.AREA)
    x_ = tf.image.resize_image_with_crop_or_pad(x_, work_h, work_w)
    with tf.variable_scope('generator'):
        construction_op = model.generator(x_, noise)

        construction_op = tf.image.resize_image_with_crop_or_pad(construction_op, s_height, s_width)

    with tf.variable_scope('upscaler'):
        upscaler_op = upscaler.generator(original_size, construction_op)

    with tf.Session() as sess:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.get_checkpoint_state(ARGS.painter_checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        ret = sess.run([upscaler_op], {x: [image]})

        ret = normalize_image(ret[0][0])
        ret = ret.astype(np.uint8)

        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ARGS.output, ret)


if __name__ == '__main__':
    main()
