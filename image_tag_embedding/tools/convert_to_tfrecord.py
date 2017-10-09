import tensorflow as tf
import os
import argparse
from datetime import datetime
import numpy as np
import pathlib
from . import util
from tflib import util as tfutil

argparser = argparse.ArgumentParser(description='Resize images that equals size of pair files')
argparser.add_argument('--vocab', type=str, help='the vocabulary file', required=True)
argparser.add_argument('--image_dir', type=str, help='the directory is contained images', required=True)
argparser.add_argument('--excludes_dir', type=str, help='the directory is contained tags', required=True)
argparser.add_argument('--tag_dir', type=str, help='the directory is contained tags', required=True)
argparser.add_argument('--out_dir', type=str, required=True)

args = argparser.parse_args()


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, max_document_length):
    tag_keys = data_set.tag_keys
    tags = data_set.tags
    images = data_set.images

    filename = pathlib.Path(args.out_dir) / (name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(str(filename))

    for index in range(len(tag_keys)):

        if index % 1000 == 0:
            print("{}: finished {}/{}".format(datetime.now(), index, len(tags)))

        tag_list = tags[tag_keys[index]]
        tag_list = np.ndarray.tolist(
            np.pad(tag_list, (0, max_document_length - len(tag_list)), 'constant').astype(np.int64))

        if not tag_keys[index] in images:
            continue

        with open(images[tag_keys[index]], 'rb') as f:
            image = f.read()

        example = tf.train.Example(features=tf.train.Features(
            feature={'tags': _int64_features(tag_list),
                     'image': _bytes_feature(image)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):

    excludes = []
    if args.excludes_dir is not None:
        excludes = tfutil.load_exclude_names(args.excludes_dir)

    vocab = util.Vocabulary()
    vocab.load(str(pathlib.Path(args.vocab)))

    tags_list = {}
    tag_keys = []
    max_tag_count = 0
    for files, ignored_files in tfutil.walk_files(args.tag_dir, excludes, 1000):
        for root, f in files:
            path = pathlib.Path(root) / f
            tag_keys.append(path.stem)

            with open(str(path)) as fp:
                tmp_tag_list = []
                for line in fp.readlines():
                    tmp_tag_list.append(int(line.strip()))

                tag_count = len(tmp_tag_list)
                max_tag_count = max(max_tag_count, tag_count)

            tags_list[path.stem] = tmp_tag_list

    image_list = {}
    for files, ignored_files in tfutil.walk_files(args.image_dir, excludes, 1000):
        for root, f in files:
            image_list[pathlib.Path(f).stem] = str(pathlib.Path(root) / f)

    if not pathlib.Path(args.out_dir).exists():
        os.makedirs(args.out_dir, mode=0o755)

    class Record(object):
        pass

    datasets = Record()
    datasets.vocab = vocab
    datasets.tag_keys = tag_keys
    datasets.tags = tags_list
    datasets.images = image_list

    convert_to(datasets, 'out', max_tag_count)
    vocab.write(str(pathlib.Path(args.out_dir) / "vocabulary.tsv"))
    print("Finish converting. Maximum tag count {}".format(max_tag_count))


if __name__ == '__main__':

    tf.app.run(main=main)
