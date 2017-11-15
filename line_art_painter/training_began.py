# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from .lib.model import model_began as model

from .lib import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=16, type=int, help='Batch size')
argparser.add_argument('--gain', default=0.001, type=int, help='propotional gain')
argparser.add_argument('--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument('--learning_rate', default=0.00001, type=float, help="learning rate[0.00001]")
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--dataset_dir', default='./datasets', type=str, help='Directory contained datasets')
argparser.add_argument('--max_steps', default=200000, type=int, help='number of maximum steps')
argparser.add_argument('--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument('--reconstruct_balance', default=0.3, type=float, help='Enable full trace of gpu')
argparser.add_argument('--log_device_placement', default=False, type=bool, help='manage logging log_device_placement')

argparser.add_argument('--balance', default=0.5, type=float, help="equilibrium balance")

ARGS = argparser.parse_args()


def train():
    with tf.Graph().as_default():

        gain = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)

        with tf.device('/cpu:0'):
            original, x = tf_dataset_input.dataset_input_fn(ARGS.dataset_dir, ARGS.batch_size)

        with tf.variable_scope('generator'):
            G = model.generator(x, 128, 128, 1, ARGS.batch_size)

        with tf.variable_scope('discriminator'):
            D = model.discriminator(original, 128, 128, 3, ARGS.batch_size)

        with tf.variable_scope('discriminator', reuse=True):
            D_G = model.discriminator(G, 128, 128, 3, ARGS.batch_size)

        g_loss = model.g_loss(G, D_G, original)
        d_loss = model.d_loss(original, D, G, D_G, gain)
        balance_d_loss = model.balanced_d_loss(original, D, G, D_G, ARGS.balance)
        measure = model.global_measure(original, D, balance_d_loss)

        tf.summary.image('base', x, max_outputs=5)
        tf.summary.image('gen', G, max_outputs=5)
        tf.summary.image('discriminated', D, max_outputs=5)
        tf.summary.image('discriminated_gen', D_G, max_outputs=5)
        tf.summary.image('original', original, max_outputs=5)

        with tf.name_scope('losses'):
            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)

        with tf.name_scope('measures'):
            tf.summary.scalar('convergence', measure)
            tf.summary.scalar('gain', gain)

        with tf.name_scope('d_train'):
            d_trainer = model.Trainer()
            d_training = d_trainer(
                d_loss,
                learning_rate=ARGS.learning_rate,
                beta1=ARGS.beta1,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

        with tf.name_scope('g_train'):
            g_trainer = model.Trainer()
            g_training = g_trainer(
                g_loss,
                learning_rate=ARGS.learning_rate,
                beta1=ARGS.beta1,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        update_gain = gain.assign(tf.clip_by_value(gain + ARGS.gain * balance_d_loss, 0, 1.0))

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

                if self._step % 10 == 0 and ARGS.full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                if self._step % 10 == 0:
                    examples_per_step = ARGS.batch_size / duration
                    measure_value = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step, measure_value, examples_per_step, sec_per_batch))

        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        update_global_step = tf.assign(global_step_tensor, global_step_tensor + 1)

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=ARGS.max_steps), tf.train.NanTensorHook(measure), _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                sess.run(
                    [d_training, g_training, update_gain, update_global_step],
                    options=run_options,
                    run_metadata=run_metadata)


if __name__ == '__main__':
    train()
