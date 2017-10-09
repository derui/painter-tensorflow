# coding: utf-8

import math
import numpy as np
import argparse
import tensorflow as tf
from . import model
import cv2

argparser = argparse.ArgumentParser(description='Generate line art from the image')
argparser.add_argument('input', type=str, help='input image')
argparser.add_argument('output', type=str, help='name of output image')
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--simplify', default=1.0, type=float, help='measure of simplification to extract of line art')

ARGS = argparser.parse_args()


def init_sess(height, width, scale):

    s_height = int(height * scale)
    s_width = int(width * scale)
    # image size to put in convolution should be able to divide 2
    work_h = int(math.pow(2, math.ceil(math.log2(height))))
    work_w = int(math.pow(2, math.ceil(math.log2(width))))
    x = tf.placeholder(tf.float32, [1, height, width, 3])
    x_ = tf.image.resize_images(x, (s_height, s_width), method=tf.image.ResizeMethod.AREA)
    x_ = tf.image.resize_images(x_, (height, width))
    x_ = tf.image.resize_image_with_crop_or_pad(x_, work_h, work_w)

    with tf.variable_scope('classifier'):
        generate_op = model.autoencoder(x_)

    generate_op = tf.image.resize_image_with_crop_or_pad(generate_op[0], height, width)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

    sess = tf.Session()
    saver = tf.train.Saver(var_list=var_list)
    ckpt = tf.train.get_checkpoint_state(ARGS.train_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    return sess, generate_op, x


def generate(sess, op, ps, image):

    ret = sess.run(op, feed_dict={ps: [image]})

    return ret


if __name__ == '__main__':
    image = cv2.imread(ARGS.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    image = image.astype(np.float32)
    image = np.multiply(image, 1 / 255.0)

    sess, op, ps = init_sess(h, w, ARGS.simplify)

    image = generate(sess, op, ps, image)

    image = np.multiply(image, 255.0)
    image = image.astype(np.uint8)

    cv2.imwrite(ARGS.output, image)
