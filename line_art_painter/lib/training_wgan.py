# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from .model import model_wgan as model

from . import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=5, type=int, help='Batch size')
argparser.add_argument('--critic_step', default=5, type=int, help='Critic steps')
argparser.add_argument('--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument('--lambda_', default=10.0, type=float, help="lambda value for gradient penalty[10.0]")
argparser.add_argument('--learning_rate', default=0.0001, type=float, help="learning rate[0.00005]")
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--dataset_dir', default='./datasets', type=str, help='Directory contained datasets')
argparser.add_argument('--max_steps', default=200000, type=int, help='number of maximum steps')
argparser.add_argument('--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument('--log_device_placement', default=False, type=bool, help='manage logging log_device_placement')

ARGS = argparser.parse_args()


def train():
    with tf.Graph().as_default():
        SIZE = 64
        DIM = SIZE * SIZE * 3

        with tf.device('/cpu:0'):
            gradient_factor = tf.random_uniform([ARGS.batch_size, 1], 0.0, 1.0)
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

            original, x = tf_dataset_input.inputs(ARGS.dataset_dir, ARGS.batch_size)
            original = tf.image.resize_images(original, (SIZE, SIZE))
            x = tf.image.resize_images(x, (SIZE, SIZE))

        with tf.variable_scope('generator'):
            G = model.generator(x)

        with tf.variable_scope('critic'):
            C = model.critic(x, original)

        with tf.variable_scope('critic', reuse=True):
            C_G = model.critic(x, G)

            _original = tf.reshape(original, [-1, DIM])
            _G = tf.reshape(G, [-1, DIM])
            penalty_image = _original + (gradient_factor * (_G - _original))
            _penalty_image = tf.reshape(penalty_image, [-1, SIZE, SIZE, 3])
            C_P = model.critic(x, _penalty_image)

        gradient_penalty = model.gradient_penalty(C_P, _penalty_image, ARGS.lambda_)
        c_loss = model.c_loss(C, C_G)
        g_loss = model.g_loss(C_G)
        l1_loss = model.l1_loss(original, G)

        tf.summary.image('base', x, max_outputs=10)
        tf.summary.image('gen', G, max_outputs=10)
        tf.summary.image('original', original, max_outputs=10)
        tf.summary.image('penalty', _penalty_image, max_outputs=10)

        with tf.name_scope('losses'):
            tf.summary.scalar("penalty", gradient_penalty)
            tf.summary.scalar("c_loss", c_loss)
            tf.summary.scalar("g_loss", g_loss)
            tf.summary.scalar("l1_loss", l1_loss)

        with tf.name_scope('c_train'):
            c_trainer = model.AdamTrainer()
            c_training = c_trainer(
                c_loss + gradient_penalty,
                beta1=ARGS.beta1,
                learning_rate=ARGS.learning_rate,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))

        with tf.name_scope('g_train'):
            g_trainer = model.AdamTrainer()
            g_training = g_trainer(
                g_loss + l1_loss,
                beta1=ARGS.beta1,
                learning_rate=ARGS.learning_rate,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                pass

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs([global_step_tensor, c_loss])

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
                    c_loss_value = run_values.results[1]
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(format_str.format(datetime.now(), self._step, c_loss_value, examples_per_step, sec_per_batch))

        update_global_step = tf.assign(global_step_tensor, global_step_tensor + 1)

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                       tf.train.NanTensorHook(c_loss), _LoggerHook()],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
                    log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                # Update generator
                sess.run([g_training], options=run_options, run_metadata=run_metadata)

                # Update critic
                for _ in range(ARGS.critic_step):
                    sess.run([c_training], options=run_options, run_metadata=run_metadata)

                sess.run([update_global_step], options=run_options, run_metadata=run_metadata)



if __name__ == '__main__':
    train()
