import tensorflow as tf
import os
import argparse

argparser = argparse.ArgumentParser(description='Resize images that equals size of pair files')
argparser.add_argument('--input_dir', type=str, help='the directory is contained images', required=True)
argparser.add_argument('--out_dir', type=str, required=True)

args = argparser.parse_args()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    painted_images = data_set.painted_images
    line_art_images = data_set.line_art_images

    filename = os.path.join(args.out_dir, name + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(data_set.num_images):

        with open(painted_images[index], 'rb') as f:
            painted_raw = f.read()
        with open(line_art_images[index], 'rb') as f:
            line_art_raw = f.read()

        example = tf.train.Example(features=tf.train.Features(
            feature={'painted': _bytes_feature(painted_raw),
                     'line_art': _bytes_feature(line_art_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):

    painted_list = []
    line_art_list = []
    for (root, _, files) in os.walk(args.input_dir):
        for f in files:
            if 'painted' in root:
                painted_list.append(os.path.join(root, f))
            else:
                line_art_list.append(os.path.join(root, f))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, mode=0o755)

    class Record(object):
        pass

    datasets = Record()
    datasets.painted_images = painted_list
    datasets.line_art_images = line_art_list
    datasets.num_images = len(painted_list)

    convert_to(datasets, 'out')


if __name__ == '__main__':

    tf.app.run(main=main)
