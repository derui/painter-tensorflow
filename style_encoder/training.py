# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from .lib.model import basic as model

from .lib import tf_dataset_input
from tflib import parameter

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=16, type=int, help='Batch size')
argparser.add_argument(
    '--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument(
    '--learning_rate',
    default=0.0001,
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
    '--max_steps', default=200000, type=int, help='number of maximum steps')
argparser.add_argument(
    '--full_trace', default=False, type=bool, help='Enable full trace of gpu')
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

ARGS = argparser.parse_args()


def train():
    with tf.Graph().as_default():

        learning_rate_v = parameter.UpdatableParameter(ARGS.learning_rate, 0.5)

        learning_rate = tf.placeholder(tf.float32, shape=[])

        with tf.device('/cpu:0'):
            iterator, (original, x) = tf_dataset_input.dataset_input_fn(
                ARGS.dataset_dir, ARGS.batch_size)

        with tf.variable_scope('style_encoder'):
            with tf.variable_scope('encoder'):
                D, pre = model.encode(original)

            with tf.variable_scope('decoder'):
                G = model.decode(D, pre)

            with tf.variable_scope('loss'):
                loss = model.loss(G, original)

            with tf.name_scope('e_train'):
                e_trainer = model.Trainer()
                e_training = e_trainer(
                    loss,
                    learning_rate=learning_rate,
                    beta1=ARGS.beta1,
                    var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='style_encoder/encoder'))

            with tf.name_scope('d_train'):
                d_trainer = model.Trainer()
                d_training = d_trainer(
                    loss,
                    learning_rate=ARGS.learning_rate,
                    beta1=ARGS.beta1,
                    var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='style_encoder/decoder'))

        tf.summary.image('gen', G, max_outputs=4)
        tf.summary.image('original', original, max_outputs=4)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time

                if self._step % 10 == 0 and ARGS.full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                if self._step % 10 == 0:
                    examples_per_step = ARGS.batch_size / duration
                    loss_value = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step,
                                          loss_value, examples_per_step,
                                          sec_per_batch))

        global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        update_global_step = tf.assign(global_step_tensor,
                                       global_step_tensor + 1)

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        lr_updater = parameter.PerEpochLossUpdater(
            learning_rate_v, steps_per_epoch=1000)

        scaffold = tf.train.Scaffold(local_init_op=tf.group(
            tf.local_variables_initializer(), iterator.initializer))
        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=ARGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                    tf.train.NanTensorHook(loss), _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                _, _, _, loss_v = sess.run(
                    [d_training, e_training, update_global_step, loss],
                    feed_dict={learning_rate: learning_rate_v()},
                    options=run_options,
                    run_metadata=run_metadata)

                # lr_updater(loss_v)


if __name__ == '__main__':
    train()
