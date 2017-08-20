# coding: utf-8

import argparse
import pathlib
import time
from datetime import datetime
from tensorflow.python.client import timeline
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from .lib.model import model
from .tools import util
from .lib import tf_dataset_input

argparser = argparse.ArgumentParser(description='Learning text embedding')
argparser.add_argument('--batch_size', default=5, type=int, help='Batch size')
argparser.add_argument('--max_document_length', type=int, required=True, help="maximum document size")
argparser.add_argument(
    '--beta1', default=0.5, type=float, help="beta1 value for optimizer [0.5]")
argparser.add_argument(
    '--learning_rate',
    default=0.0001,
    type=float,
    help="learning rate[0.00005]")
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
    '--log_device_placement',
    default=False,
    type=bool,
    help='manage logging log_device_placement')
argparser.add_argument('--vocab', type=str, help="vocabulary file")

ARGS = argparser.parse_args()

EMBEDDING_SIZE = 300


def train():

    vocab = util.Vocabulary()
    vocab.load(str(pathlib.Path(ARGS.vocab)))

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

            tags, images = tf_dataset_input.inputs(ARGS.dataset_dir, ARGS.batch_size,
                                                   ARGS.max_document_length)
            embedding = tf.Variable(
                tf.random_uniform([len(vocab), EMBEDDING_SIZE], -1.0, 1.0),
                name="embedding")
            lookupped = tf.nn.embedding_lookup(embedding, tags)
            lookupped = tf.expand_dims(lookupped, -1)

        E = model.embedding_encoder(lookupped)
        D = model.image_autoencoder(images, E)

        loss = model.loss(D, images)

        tf.summary.image("original", images)
        tf.summary.image("decoded", D)
        tf.summary.scalar("loss", loss)

        with tf.name_scope('c_train'):
            training = model.training(loss, ARGS.learning_rate,
                                      global_step_tensor, None)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime """

            def begin(self):
                pass

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs([global_step_tensor, loss])

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
                    loss_value = run_values.results[1]
                    sec_per_batch = float(duration)

                    format_str = '{}: step {}, loss = {:.3f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(
                        format_str.format(datetime.now(), self._step,
                                          loss_value, examples_per_step,
                                          sec_per_batch))

        run_options = tf.RunOptions()
        if ARGS.full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        metadata_path = write_metadata(vocab)
        sliced_embedding = tf.Variable(
            tf.random_uniform([500, EMBEDDING_SIZE], -1.0, 1.0),
            trainable=False,
            name="sliced_embedding")
        assign = tf.assign(sliced_embedding, embedding[:500])

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ARGS.train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=ARGS.max_steps),
                    tf.train.NanTensorHook(loss), _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(
                        per_process_gpu_memory_fraction=0.85),
                    log_device_placement=ARGS.log_device_placement)) as sess:

            while not sess.should_stop():

                sess.run([training], options=run_options, run_metadata=run_metadata)

                step = sess.run(global_step_tensor)
                if step > 0 and step % 100 == 0:
                    sess.run(assign)
                    config = projector.ProjectorConfig()
                    emb = config.embeddings.add()
                    emb.tensor_name = sliced_embedding.name
                    emb.metadata_path = str(metadata_path.absolute())

                    summary_writer = tf.summary.FileWriter(ARGS.train_dir)
                    projector.visualize_embeddings(summary_writer, config)


def write_metadata(vocab):
    vocabs = [v for v in vocab.retrieve()]
    vocab_list = sorted(vocabs, key=lambda v: v[1])
    vocab_list = vocab_list[:500]

    metadata_path = pathlib.Path(ARGS.train_dir) / "metadata.tsv"

    if not metadata_path.parent.exists():
        metadata_path.parent.mkdir(parents=True)

    with open(str(metadata_path), "w") as f:
        f.write("Name\tFreq\n")
        for (tag, _, freq) in vocab_list:
            f.write("{}\t{}\n".format(tag, freq))

    return metadata_path


if __name__ == '__main__':
    train()
