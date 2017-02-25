# coding: utf-8

import argparse
import time
from datetime import datetime
import tensorflow as tf
import model

import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning painter model')
argparser.add_argument('--batch_size', default=5, type=int, help='Batch size')
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

ARGS = argparser.parse_args()


def train():

    with tf.Graph().as_default():
        global_step_tensor = tf.Variable(10, trainable=False, name='global_step')

        original, x = tf_dataset_input.inputs(ARGS.dataset_dir,
                                              ARGS.batch_size)

        construction_op = model.generator(x, 512, 512, 3)
        loss_op = model.loss(original, construction_op, x)
        training_op = model.training(loss_op, 0.05)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss_op)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results

                if self._step % 10 == 0:
                    examples_per_step = ARGS.batch_size / duration
                    loss_value = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.2} ({:.1} examples/sec; {:.3} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step,
                                          loss_value, examples_per_step,
                                          sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                    tf.train.NanTensorHook(loss_op), _LoggerHook()
                ]) as sess:
            tf.train.global_step(sess, global_step_tensor)
            while not sess.should_stop():
                sess.run(training_op)


if __name__ == '__main__':
    train()
