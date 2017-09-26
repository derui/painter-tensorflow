# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from .lib.model import model_sd_began as model

from .lib import tf_dataset_input_sd_began as tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=2, type=int, help='Batch size')
argparser.add_argument('--critic_step', default=2, type=int, help='Critic steps')
argparser.add_argument('--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument('--gain', default=0.001, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument('--balance', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument('--learning_rate', default=0.0001, type=float, help="learning rate[0.00005]")
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--dataset_dir', default='./datasets', type=str, help='Directory contained datasets')
argparser.add_argument('--max_steps', default=200000, type=int, help='number of maximum steps')
argparser.add_argument('--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument('--log_device_placement', default=False, type=bool, help='manage logging log_device_placement')

ARGS = argparser.parse_args()

NOISE_SIZE = 64
SIZE = 128
DIM = SIZE * SIZE * 3


def train():

    with tf.Graph().as_default():
        gain = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)

        with tf.device('/cpu:0'):
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            noise_base = tf.random_uniform([ARGS.batch_size, NOISE_SIZE],
                                           minval=-1.0, maxval=1.0,
                                           dtype=tf.float32)

            z0 = tf.random_uniform([ARGS.batch_size, NOISE_SIZE],
                                   minval=-1.0, maxval=1.0,
                                   dtype=tf.float32)

            original, x = tf_dataset_input.inputs(ARGS.dataset_dir, ARGS.batch_size, distorted=False)
            original = tf.image.resize_images(original, (SIZE, SIZE))
            x = tf.image.resize_images(x, (SIZE, SIZE))

        with tf.variable_scope('generator'):
            G = model.generator(x, tf.concat([noise_base, z0], 1))

        with tf.variable_scope('discriminator'):
            D_original = model.discriminator(original, x)

        with tf.variable_scope('discriminator', reuse=True):
            D = model.discriminator(G, x)

        d_loss = model.d_loss(original, D_original, G, D, gain)
        balance_d_loss = model.balanced_d_loss(original, D_original, G, D, ARGS.balance)
        g_loss = model.g_loss(G, D, original)

        tf.summary.image('base', x, max_outputs=10)
        tf.summary.image('original', original, max_outputs=10)
        tf.summary.image('gen', G, max_outputs=10)
        tf.summary.image('encoded_original', D_original, max_outputs=10)
        tf.summary.image('encoded_gen', D, max_outputs=10)

        with tf.name_scope('losses'):
            tf.summary.scalar("d_loss", d_loss)
            tf.summary.scalar("g_loss", g_loss)
            tf.summary.scalar("gain", gain)

        with tf.name_scope('d_train'):
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_trainer = model.AdamTrainer()
            d_training = d_trainer(
                d_loss,
                beta1=ARGS.beta1,
                learning_rate=ARGS.learning_rate,
                var_list=d_vars)

        with tf.name_scope('g_train'):
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

            g_trainer = model.AdamTrainer()
            g_training = g_trainer(
                g_loss,
                beta1=ARGS.beta1,
                learning_rate=ARGS.learning_rate,
                var_list=g_vars)

        update_gain = gain.assign(tf.clip_by_value(gain + ARGS.gain * balance_d_loss, 0, 1.0))

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                pass

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs([global_step_tensor, d_loss])

            def after_run(self, run_context, run_values):
                self._step = run_values.results[0]
                duration = time.time() - self._start_time

                if self._step % 10 == 0 and ARGS.full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                if self._step % 10 == 0:
                    examples_per_step = ARGS.batch_size / duration
                    d_loss_value = run_values.results[1]
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(format_str.format(datetime.now(), self._step,
                                            d_loss_value,
                                            examples_per_step, sec_per_batch))

        update_global_step = tf.assign(global_step_tensor, global_step_tensor + 1)

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                       tf.train.NanTensorHook(d_loss),
                       _LoggerHook()],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
                    log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                sess.run([d_training, g_training, update_gain, update_global_step],
                         options=run_options, run_metadata=run_metadata)


if __name__ == '__main__':
    train()
