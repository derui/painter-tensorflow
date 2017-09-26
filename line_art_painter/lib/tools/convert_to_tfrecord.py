import pathlib
import tensorflow as tf
import os
import random
import argparse
from datetime import datetime
import numpy as np

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
argparser.add_argument('--max_document_length', type=int, required=True)
argparser.add_argument('--validation_size', type=float, default=0.3)

args = argparser.parse_args()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert_to(data_set, name, max_document_length):
    filename = pathlib.Path(args.out_dir) / (name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(str(filename))

    num_images = len(data_set.data)
    for index in range(num_images):

        if index % 1000 == 0 and index > 0:
            print("{}: finished {}/{}".format(datetime.now(), index,
                                              num_images))

        with open(data_set.data[index]['original'], 'rb') as f:
            original_raw = f.read()
        with open(data_set.data[index]['line_art'], 'rb') as f:
            line_art_raw = f.read()
        with open(data_set.data[index]['tag']) as f:
            tag_list = list(map(lambda x: x.strip(), f.readlines()))
            tag_list = list(map(lambda x: int(x), tag_list))
            tag_list = np.ndarray.tolist(
                np.pad(tag_list, (0, max_document_length - len(tag_list)),
                       'constant').astype(np.int64))

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
            original_list.append(str(pathlib.Path(root) / f))

    for (root, _, files) in os.walk(args.line_art_dir):
        for f in files:
            line_art_list.append(str(pathlib.Path(root) / f))

    for (root, _, files) in os.walk(args.tag_dir):
        for f in files:
            tags_list.append(str(pathlib.Path(root) / f))

    if not pathlib.Path(args.out_dir).exists():
        os.makedirs(args.out_dir, mode=0o755)

    class Record(object):
        pass

    datasets = Record()
    datasets.original_images = sorted(original_list)
    datasets.line_art_images = sorted(line_art_list)
    datasets.tags = sorted(tags_list)
    datasets.tag_classes = 1000
    datasets.num_images = len(original_list)

    merged_dataset = []
    train_set = Record()
    train_set.data = []
    validation_set = Record()
    validation_set.data = []

    for i in range(len(original_list)):
        merged_dataset.append({
            'original': datasets.original_images[i],
            'line_art': datasets.line_art_images[i],
            'tag': datasets.tags[i],
        })

    random.shuffle(merged_dataset)
    validation_num = int(args.validation_size * len(merged_dataset))
    train_set.data = merged_dataset[validation_num:]
    validation_set.data = merged_dataset[:validation_num]

    convert_to(train_set, 'train', args.max_document_length)
    convert_to(validation_set, 'validation', args.max_document_length)


if __name__ == '__main__':

    tf.app.run(main=main)
