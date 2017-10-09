# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf
from . import model

from .lib import tf_dataset_input as dataset
from tflib import parameter

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=15, type=int, help='Batch size')
argparser.add_argument('--learning_rate', default=0.0002, type=float, help="learning rate[0.0002]")
argparser.add_argument('--dataset_dir', type=str, required=True, help='Directory contained datasets')
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--max_steps', default=20000, type=int, help='number of maximum steps')
argparser.add_argument('--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument('--log_device_placement', default=False, type=bool, help='manage logging log_device_placement')
argparser.add_argument('--image_size', default=128, type=int, help='size of training image contained dataset')
argparser.add_argument('--alpha', default=6, type=int)
argparser.add_argument('--beta', default=-2, type=int)
argparser.add_argument('--bins', default=10, type=int)
argparser.add_argument('--distorted', default=True, type=bool)

ARGS = argparser.parse_args()


def train():

    with tf.Graph().as_default():
        learning_rate_v = parameter.UpdatableParameter(ARGS.learning_rate, 0.1)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        with tf.device('/cpu:0'):
            painted, line_art = dataset.dataset_input_fn(
                ARGS.dataset_dir, ARGS.batch_size, ARGS.image_size, distorted=ARGS.distorted)

        with tf.variable_scope('classifier'):
            encoded = model.autoencoder(painted)

        tf.summary.image('encoded', encoded)
        tf.summary.image('painted', painted)
        tf.summary.image('line_art', line_art)

        lmap = model.loss_map(line_art, ARGS.bins, ARGS.alpha, ARGS.beta)
        loss = model.loss(line_art, encoded, lmap)

        training = model.training(loss, learning_rate=learning_rate, global_step=global_step_tensor, var_list=None)

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

                    format_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(format_str.format(datetime.now(), self._step, loss_value, examples_per_step, sec_per_batch))

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        lr_updater = parameter.PerEpochLossUpdater(learning_rate_v, steps_per_epoch=1000)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[tf.train.StopAtStepHook(num_steps=ARGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
                    log_device_placement=ARGS.log_device_placement)) as sess:
            while not sess.should_stop():
                _, loss_v = sess.run([training, loss], options=run_options, run_metadata=run_metadata)

                lr_updater(loss_v)


if __name__ == '__main__':
    train()
