import tensorflow as tf
import os
import argparse
from datetime import datetime

argparser = argparse.ArgumentParser(
    description='Resize images that equals size of pair files')
argparser.add_argument(
    '--original_dir',
    type=str,
    help='the directory is contained images',
    required=True)
argparser.add_argument(
    '--tag_dir',
    type=str,
    help='the directory is contained tags',
    required=True)
argparser.add_argument(
    '--line_art_dir',
    type=str,
    help='the directory is contained images',
    required=True)
argparser.add_argument('--out_dir', type=str, required=True)

args = argparser.parse_args()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to(data_set, name):
    original_images = data_set.original_images
    line_art_images = data_set.line_art_images
    tags = data_set.tags

    filename = os.path.join(args.out_dir, name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(data_set.num_images):

        if index % 1000 == 0:
            print("{}: finished {}/{}".format(datetime.now(), index, data_set.num_images))

        with open(original_images[index], 'rb') as f:
            original_raw = f.read()
        with open(line_art_images[index], 'rb') as f:
            line_art_raw = f.read()
        with open(tags[index]) as f:
            tag_list = f.read().split(',')
            tag_list = list(map(lambda x: int(x), tag_list))

        example = tf.train.Example(features=tf.train.Features(feature={
            'original': _bytes_feature(original_raw),
            'line_art': _bytes_feature(line_art_raw),
            'tags': _int64_features(tag_list)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):

    original_list = []
    line_art_list = []
    tags_list = []
    for (root, _, files) in os.walk(args.original_dir):
        for f in files:
            original_list.append(os.path.join(root, f))

    for (root, _, files) in os.walk(args.line_art_dir):
        for f in files:
            line_art_list.append(os.path.join(root, f))

    for (root, _, files) in os.walk(args.tag_dir):
        for f in files:
            tags_list.append(os.path.join(root, f))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, mode=0o755)

    class Record(object):
        pass

    datasets = Record()
    datasets.original_images = sorted(original_list)
    datasets.line_art_images = sorted(line_art_list)
    datasets.tags = sorted(tags_list)
    datasets.tag_classes = 1000
    datasets.num_images = len(original_list)

    convert_to(datasets, 'out')


if __name__ == '__main__':

    tf.app.run(main=main)
