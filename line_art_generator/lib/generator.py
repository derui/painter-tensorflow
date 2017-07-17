# coding: utf-8

import os
import math
import numpy as np
import argparse
import tensorflow as tf
import concurrent.futures
from .model import model
import cv2


def init_sess(batch_size, height, width, train_dir, reuse=False):

    image_size = int(math.pow(2, math.ceil(math.log2(max(height, width)))))
    x = tf.placeholder(tf.float32, [batch_size, height, width, 3])

    with tf.variable_scope('classifier', reuse=reuse):
        generate_op = model.autoencoder(x)

    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

    sess = tf.Session()
    saver = tf.train.Saver(var_list=var_list)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    return sess, generate_op, x


def generate(sess, op, ps, images):

    ret = sess.run(op, feed_dict={ps: images})

    return ret


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Generate line art from the image')
    argparser.add_argument('input_dir', type=str, help='input image')
    argparser.add_argument('output_dir', type=str, help='name of output image')
    argparser.add_argument(
        '--train_dir',
        default='./log',
        type=str,
        help='Directory will have been saving checkpoint')
    argparser.add_argument(
        '--image_size',
        default=128,
        type=int)
    argparser.add_argument(
        '--batch_size',
        default=30,
        type=int)

    ARGS = argparser.parse_args()

    def write_images(images, output_dir):
        def writer(img, infile, output_dir):
            fname = os.path.basename(infile)

            img = np.multiply(img, 255.0)
            img = img.astype(np.uint8)

            if not os.path.exists(os.path.join(output_dir, fname[:2])):
                os.makedirs(os.path.join(output_dir, fname[:2]), exist_ok=True)

            cv2.imwrite(os.path.join(output_dir, fname[:2], fname), img)

        with concurrent.futures.ThreadPoolExecutor(16) as e:
            futures = []

            for img, infile in images:
                futures.append(e.submit(writer, img, infile, output_dir))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print('exception: %s' % exc)

    if not os.path.exists(ARGS.output_dir):
        os.makedirs(ARGS.output_dir, exist_ok=True)

    input_files = []
    for (root, _, files) in os.walk(ARGS.input_dir):
        for f in files:
            input_files.append(os.path.join(root, f))

    input_files = [input_files[i:i + ARGS.batch_size] for i in range(0, len(input_files), ARGS.batch_size)]
    rest_input_files = input_files[-1]
    input_files = input_files[:-1]

    sess, op, ps = init_sess(ARGS.batch_size, ARGS.image_size, ARGS.image_size,
                             ARGS.train_dir)
    for i in range(len(input_files)):
        files = input_files[i]
        images = [cv2.imread(f) for f in files]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = [image.astype(np.float32) for image in images]
        images = [np.multiply(image, 1 / 255.0) for image in images]

        images = generate(sess, op, ps, images)
        write_images(zip(images, files), ARGS.output_dir)
        print("Finished batch {}".format(i+1))

    sess.close()

    sess, op, ps = init_sess(len(rest_input_files), ARGS.image_size, ARGS.image_size,
                             ARGS.train_dir,
                             reuse=True)
    images = [cv2.imread(f) for f in rest_input_files]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = [image.astype(np.float32) for image in images]
    images = [np.multiply(image, 1 / 255.0) for image in images]

    images = generate(sess, op, ps, images)

    write_images(zip(images, rest_input_files), ARGS.output_dir)
