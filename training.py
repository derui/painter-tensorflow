# coding: utf-8

import os
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

    with tf.name_scope('d_train'):
        d_trainer = model.Trainer()
        d_training = d_trainer(
            d_loss,
            learning_rate=ARGS.learning_rate,
            beta1=ARGS.beta1,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    with tf.name_scope('g_train'):
        g_trainer = model.Trainer()
        g_training = g_trainer(
            g_loss,
            learning_rate=ARGS.learning_rate,
            beta1=ARGS.beta1,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

    class LoggingSession(object):
        """Logs loss and runtime """

        def __init__(self,
                     sess,
                     train_dir,
                     max_steps,
                     save_summary_per_step=100,
                     full_trace=False):
            self._step = -1
            self.sess = sess
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(train_dir)
            self.max_steps = max_steps
            self.train_dir = train_dir
            self.full_trace = full_trace
            self._checkpoint_time = time.time()
            self.save_checkpoint_per_sec = 60
            self.merged_summaries = tf.summary.merge_all()
            self._save_summary_per_step = save_summary_per_step

        def finish_session(self):
            self.save_summary()
            self.save_checkpoint(self.max_steps)

        def save_checkpoint(self, steps):
            self.saver.save(
                self.sess,
                os.path.join(self.train_dir, 'model.ckpt'),
                global_step=steps)

        def save_summary(self):
            summary = self.sess.run(self.merged_summaries)
            self.summary_writer.add_summary(summary,
                                            tf.train.get_global_step())

        def _restore_if_exists(self):
            ckpt = tf.train.get_checkpoint_state(self.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        def run(self):
            self._restore_if_exists()

            step_wrote_summaries = self._save_summary_per_step
            self._checkpoint_time = time.time()
            for i in range(self.max_steps):
                args = self.before_run()

                # run training operations.
                self.sess.run(
                    d_training, options=run_options, run_metadata=run_metadata)

                self.sess.run(
                    g_training, options=run_options, run_metadata=run_metadata)

                results = sess.run(args)
                self.after_run(results)

                if i >= step_wrote_summaries:
                    self.save_summary()
                    step_wrote_summaries = i + self._save_summary_per_step

                duration = time.time() - self._checkpoint_time
                if int(duration) > self.save_checkpoint_per_sec:
                    self._checkpoint_time = time.time()
                    self.save_checkpoint(i)

        def before_run(self):
            self._step += 1
            self._start_time = time.time()
            return [d_loss, g_loss]

        def after_run(self, run_values):
            duration = time.time() - self._start_time

            if self._step % 10 == 0 and self.full_trace:
                # write train
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)

            if self._step % 10 == 0:
                examples_per_step = ARGS.batch_size / duration
                d_loss_value, g_loss_value = run_values
                sec_per_batch = float(duration)

                format_str = '{}: step {}, loss = {:.2f},{:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                print(
                    format_str.format(datetime.now(), self._step, d_loss_value,
                                      g_loss_value, examples_per_step,
                                      sec_per_batch))

    init_op = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(
                log_device_placement=ARGS.log_device_placement)) as sess:

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(
                    qr.create_threads(
                        sess, coord=coord, daemon=True, start=True))

            logging_session = LoggingSession(
                sess,
                ARGS.train_dir,
                ARGS.max_steps,
                full_trace=ARGS.full_trace)

            logging_session.run()

            logging_session.finish_session()

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    train()
