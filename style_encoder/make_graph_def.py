# coding: utf-8

from pathlib import Path
import argparse
import tensorflow as tf
from .lib.model import basic as model

DEFAULT_IMAGE_SIZE = 128


def create_argparser():
    argparser = argparse.ArgumentParser(description='Make encoder graph')
    argparser.add_argument(
        '--input_checkpoint',
        type=str,
        help='Directory will have been saving checkpoint')
    argparser.add_argument(
        '--output_checkpoint',
        type=str,
        help='Location to output converted graph_def')

    return argparser


def train(args):
    with tf.Graph().as_default():

        config = tf.ConfigProto(
            log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            input_checkpoint = Path(args.input_checkpoint)
            checkpoint = tf.train.latest_checkpoint(str(input_checkpoint))
            input_checkpoint = Path(checkpoint)

            saver = tf.train.import_meta_graph(
                str(input_checkpoint) + ".meta", clear_devices=True)
            saver.restore(sess, str(input_checkpoint))
            var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope='style_encoder/encoder')

            names = [v.name[:-2] for v in var_list]
            print(names)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, tf.get_default_graph().as_graph_def(), names)

            output_checkpoint = Path(args.output_checkpoint)
            if not output_checkpoint.exists():
                output_checkpoint.mkdir(parents=True)
            with tf.gfile.GFile(
                    str(Path(output_checkpoint, "model.pb")), "wb") as f:
                f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    argparser = create_argparser()
    args = argparser.parse_args()
    train(args)
