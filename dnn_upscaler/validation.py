# coding: utf-8

import argparse
import time
from datetime import datetime
from tensorflow.python.client import timeline
import tensorflow as tf

from .lib.model import basic as model
from .lib import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning supervised painting model')
argparser.add_argument('--batch_size', default=5, type=int, help='Batch size')
argparser.add_argument('--train_dir', default='./log', type=str, help='Directory will have been saving checkpoint')
argparser.add_argument('--dataset_dir', default='./datasets', type=str, help='Directory contained datasets')
argparser.add_argument('--max_steps', default=200000, type=int, help='number of maximum steps')
argparser.add_argument('--full_trace', default=False, type=bool, help='Enable full trace of gpu')
argparser.add_argument('--log_device_placement', default=False, type=bool, help='manage logging log_device_placement')

ARGS = argparser.parse_args()


def varidation():
    with tf.Graph().as_default():
        SIZE = 512

        with tf.device('/cpu:0'):
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

            original = tf_dataset_input.dataset_input_fn(ARGS.dataset_dir, "validation.tfrecords", ARGS.batch_size, SIZE)
            small = tf.image.resize_images(original, (SIZE // 4, SIZE // 4), method=tf.image.ResizeMethod.AREA)

        with tf.variable_scope('upsampler'):
            S = model.upsampler(small)

        l1_loss = model.l1_loss(original, S)

        tf.summary.image('upscaled', S, max_outputs=10)
        tf.summary.image('small', small, max_outputs=10)
        tf.summary.image('original', original, max_outputs=10)

        with tf.name_scope('losses'):
            tf.summary.scalar("l1_loss", l1_loss)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                self.step = 0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs([l1_loss])

            def after_run(self, run_context, run_values):
                self.step += 1
                duration = time.time() - self._start_time

                if self.step % 10 == 0 and ARGS.full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                if self.step % 10 == 0:
                    examples_per_step = ARGS.batch_size / duration
                    [c_loss_value] = run_values.results
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(format_str.format(datetime.now(), self.step, c_loss_value, examples_per_step, sec_per_batch))

        update_global_step = tf.assign(global_step_tensor, global_step_tensor + 1)

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        with tf.train.MonitoredTrainingSession(
                save_checkpoint_secs=None,
                checkpoint_dir=ARGS.train_dir,
                hooks=[tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                       tf.train.NanTensorHook(l1_loss), _LoggerHook()],
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
                    log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                sess.run([l1_loss, update_global_step], options=run_options, run_metadata=run_metadata)


if __name__ == '__main__':
    varidation()
