# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from model import model

import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=5, type=int, help='Batch size')
argparser.add_argument(
    '--learning_rate',
    default=0.0002,
    type=float,
    help="learning rate[0.0002]")
argparser.add_argument(
    '--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
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
    '--max_steps', default=20000, type=int, help='number of maximum steps')
argparser.add_argument(
    '--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument(
    '--log_device_placement',
    default=False,
    type=bool,
    help='manage logging log_device_placement')

ARGS = argparser.parse_args()


def train():

    with tf.Graph().as_default():
        global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')

        with tf.device('/cpu:0'):
            original, x = tf_dataset_input.inputs(ARGS.dataset_dir,
                                                  ARGS.batch_size)

        with tf.variable_scope('generator'):
            G = model.generator(x, 128, 128, 3, ARGS.batch_size)

        tf.summary.image('base', x, max_outputs=10)
        tf.summary.image('gen', G, max_outputs=10)
        tf.summary.image('origin', original, max_outputs=10)

        with tf.variable_scope('discriminator'):
            D = model.discriminator(original, 128, 128, 3)

        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            D_G = model.discriminator(G, 128, 128, 3)

        d_loss = model.d_loss(D, D_G)
        g_loss = model.g_loss(D_G)

        d_training = model.training(
            d_loss,
            learning_rate=ARGS.learning_rate,
            beta1=ARGS.beta1,
            global_step=global_step_tensor,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

        g_training = model.training(
            g_loss,
            learning_rate=ARGS.learning_rate,
            beta1=ARGS.beta1,
            global_step=global_step_tensor,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs([d_loss, g_loss])

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
                    d_loss_value, g_loss_value = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.2f},{:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step,
                                          d_loss_value, g_loss_value,
                                          examples_per_step, sec_per_batch))

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                    tf.train.NanTensorHook(d_loss),
                    tf.train.NanTensorHook(g_loss), _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    log_device_placement=ARGS.log_device_placement)) as sess:
            tf.train.global_step(sess, global_step_tensor)
            while not sess.should_stop():
                sess.run(
                    d_training, options=run_options, run_metadata=run_metadata)
                sess.run(
                    g_training, options=run_options, run_metadata=run_metadata)
                sess.run(
                    g_training, options=run_options, run_metadata=run_metadata)


if __name__ == '__main__':
    train()
