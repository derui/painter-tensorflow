# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from .lib.model import model_began as model

from .lib import tf_dataset_input
from tflib import parameter
from style_encoder.lib.model import basic as style_encoder


def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Learning painter model')
    argparser.add_argument(
        '--batch_size', default=16, type=int, help='Batch size')
    argparser.add_argument(
        '--beta1',
        default=0.5,
        type=float,
        help="beta1 value for optimizer [0.5]")
    argparser.add_argument(
        '--learning_rate',
        default=0.00001,
        type=float,
        help="learning rate[0.00001]")
    argparser.add_argument(
        '--train_dir',
        default='./log',
        type=str,
        help='Directory will have been saving checkpoint')
    argparser.add_argument(
        '--dataset_dir',
        default='./datasets',
        type=str,
        help='Directory contained datasets')
    argparser.add_argument(
        '--max_steps',
        default=200000,
        type=int,
        help='number of maximum steps')
    argparser.add_argument(
        '--full_trace',
        default=False,
        type=bool,
        help='Enable full trace of gpu')
    argparser.add_argument(
        '--reconstruct_balance',
        default=0.3,
        type=float,
        help='Enable full trace of gpu')
    argparser.add_argument(
        '--log_device_placement',
        default=False,
        type=bool,
        help='manage logging log_device_placement')

    argparser.add_argument(
        '--balance', default=0.5, type=float, help="equilibrium balance")
    argparser.add_argument(
        '--style_encoder_graph',
        type=str,
        help="The file which has frozen graph for style encoder")

    return argparser


def load_style_encoder_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def train(args):
    style_encoder_graph = load_style_encoder_graph(args.style_encoder_graph)
    with tf.Graph().as_default():
        tf.import_graph_def(style_encoder_graph)

        learning_rate_v = parameter.UpdatableParameter(args.learning_rate, 0.5)

        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        gain = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)

        with tf.device('/cpu:0'):
            iterator, (original, x) = tf_dataset_input.dataset_input_fn(
                args.dataset_dir, args.batch_size, validation=True)

        with tf.variable_scope('style_encoder/encoder'):
            style, _ = style_encoder.encode(original)

        with tf.variable_scope('generator'):
            G = model.generator(x, style)

        with tf.variable_scope('discriminator'):
            D = model.discriminator(original)

        with tf.variable_scope('discriminator', reuse=True):
            D_G = model.discriminator(G)

        g_loss = model.g_loss(G, D_G, original)
        d_loss = model.d_loss(original, D, G, D_G, gain)
        balance_d_loss = model.balanced_d_loss(original, D, G, D_G,
                                               args.balance)
        measure = model.global_measure(original, D, balance_d_loss)

        tf.summary.image('base', x, max_outputs=4)
        tf.summary.image('gen', G, max_outputs=4)
        tf.summary.image('discriminated', D, max_outputs=4)
        tf.summary.image('discriminated_gen', D_G, max_outputs=4)
        tf.summary.image('original', original, max_outputs=4)

        with tf.name_scope('losses'):
            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)

        with tf.name_scope('measures'):
            tf.summary.scalar('convergence', measure)
            tf.summary.scalar('gain', gain)
            tf.summary.scalar('learning_rate', learning_rate)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(measure)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time

                if self._step % 10 == 0 and args.full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                if self._step % 10 == 0:
                    examples_per_step = args.batch_size / duration
                    measure_value = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step,
                                          measure_value, examples_per_step,
                                          sec_per_batch))

        global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        update_global_step = tf.assign(global_step_tensor,
                                       global_step_tensor + 1)

        run_options = tf.RunOptions()
        if args.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        lr_updater = parameter.PerEpochLossUpdater(
            learning_rate_v, steps_per_epoch=1000)

        scaffold = tf.train.Scaffold(local_init_op=tf.group(
            tf.local_variables_initializer(), iterator.initializer),
                                     init_feed_dict={learning_rate: learning_rate_v()})
        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=args.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=args.max_steps),
                    tf.train.NanTensorHook(measure), _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    log_device_placement=args.log_device_placement)) as sess:

            while not sess.should_stop():

                sess.run(
                    [update_global_step, g_loss],
                    feed_dict={learning_rate: learning_rate_v()},
                    options=run_options,
                    run_metadata=run_metadata)

                # lr_updater(loss_v)


if __name__ == '__main__':
    argparser = create_arg_parser()
    args = argparser.parse_args()

    train(args)
