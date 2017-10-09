import os
import argparse
import random
from datetime import datetime

import tensorflow as tf

argparser = argparse.ArgumentParser(description='Resize images')
argparser.add_argument('--image_dir', type=str, help='the directory is contained images', required=True)
argparser.add_argument('--out_dir', type=str, required=True)
argparser.add_argument('--validation_size', type=float, default=0.3)

args = argparser.parse_args()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    original_images = data_set.data

    filename = os.path.join(args.out_dir, name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(data_set.num_images):

        if index > 0 and index % 1000 == 0:
            print("{}: finished {}/{}".format(datetime.now(), index, data_set.num_images))

        with open(original_images[index], 'rb') as f:
            original_raw = f.read()

        example = tf.train.Example(features=tf.train.Features(feature={'original': _bytes_feature(original_raw), }))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):

    original_list = []
    for (root, _, files) in os.walk(args.image_dir):
        for f in files:
            original_list.append(os.path.join(root, f))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, mode=0o755)

    class Record(object):
        pass

    train_set = Record()
    validation_set = Record()
    random.shuffle(original_list)
    validation_num = int(args.validation_size * len(original_list))
    train_set.data = original_list[validation_num:]
    train_set.num_images = len(train_set.data)
    validation_set.data = original_list[:validation_num]
    validation_set.num_images = len(validation_set.data)

    convert_to(train_set, 'train')
    convert_to(validation_set, 'validation')


if __name__ == '__main__':

    tf.app.run(main=main)
