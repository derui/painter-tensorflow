# coding: utf-8

from pathlib import Path
import numpy as np
import argparse
import time
import pickle

from tensorflow.python.client import timeline
import tensorflow as tf

import GPyOpt

from .lib.model import model_began as model
from .lib import tf_dataset_input
from tflib import parameter
from style_encoder.lib.model import basic as style_encoder


def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Learning painter model')
    argparser.add_argument(
        '--batch_size', default=16, type=int, help='Batch size')
    argparser.add_argument(
        '--gain', default=0.001, type=int, help='propotional gain')
    argparser.add_argument(
        '--beta1',
        default=0.5,
        type=float,
        help="beta1 value for optimizer [0.5]")
    argparser.add_argument(
        '--learning_rate',
        default=0.00001,
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
        '--max_steps',
        default=200000,
        type=int,
        help='number of maximum steps')
    argparser.add_argument(
        '--full_trace',
        default=False,
        type=bool,
        help='Enable full trace of gpu')
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

    argparser.add_argument(
        '--balance', default=0.5, type=float, help="equilibrium balance")
    argparser.add_argument(
        '--style_encoder_graph',
        type=str,
        help="The file which has frozen graph for style encoder")

    return argparser


def load_style_encoder_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def run_began(style_encoder_graph, dataset_dir, train_dir, learning_rate,
              beta1, batch_size, balance, gain_balance, max_steps,
              log_device_placement, full_trace):

    style_encoder_graph = load_style_encoder_graph(style_encoder_graph)
    with tf.Graph().as_default():
        tf.import_graph_def(style_encoder_graph)

        learning_rate_v = parameter.UpdatableParameter(learning_rate, 0.5)

        learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate")
        gain = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)

        with tf.device('/cpu:0'):
            iterator, (original, x) = tf_dataset_input.dataset_input_fn(
                dataset_dir, batch_size)
            gray_original = tf.image.rgb_to_grayscale(original)

        with tf.variable_scope('style_encoder/encoder'):
            style, _ = style_encoder.encode(original)

        with tf.variable_scope('generator'):
            prev_style, after_style, G = model.generator(x, style)

        with tf.variable_scope('guide_decoder'):
            G1 = model.guide_decoder1(prev_style)
            G2 = model.guide_decoder2(after_style)

        with tf.variable_scope('discriminator'):
            D = model.discriminator(original)

        with tf.variable_scope('discriminator', reuse=True):
            D_G = model.discriminator(G)

        g_loss = model.g_loss(G, D_G, original, G1, G2, gray_original)
        d_loss = model.d_loss(original, D, G, D_G, gain)
        balance_d_loss = model.balanced_d_loss(original, D, G, D_G, balance)
        measure = model.global_measure(original, D, balance_d_loss)

        tf.summary.image('base', x, max_outputs=2)
        tf.summary.image('gen', G, max_outputs=2)
        tf.summary.image('discriminated', D, max_outputs=2)
        tf.summary.image('discriminated_gen', D_G, max_outputs=2)
        tf.summary.image('original', original, max_outputs=2)
        tf.summary.image('guide1', G1, max_outputs=2)
        tf.summary.image('guide2', G2, max_outputs=2)

        with tf.name_scope('losses'):
            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('g_loss', g_loss)

        with tf.name_scope('measures'):
            tf.summary.scalar('convergence', measure)
            tf.summary.scalar('gain', gain)
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('d_train'):
            d_trainer = model.Trainer()
            d_training = d_trainer(
                d_loss,
                learning_rate=learning_rate,
                beta1=beta1,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

        with tf.name_scope('g_train'):
            var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            var_list.extend(
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='guide_decoder'))
            g_trainer = model.Trainer()
            g_training = g_trainer(
                g_loss,
                learning_rate=learning_rate,
                beta1=beta1,
                var_list=var_list)

        update_gain = gain.assign(
            tf.clip_by_value(gain + gain_balance * balance_d_loss, 0, 1.0))

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

                if self._step % 10 == 0 and full_trace:
                    # write train
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

        global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        update_global_step = tf.assign(global_step_tensor,
                                       global_step_tensor + 1)

        run_options = tf.RunOptions()
        if full_trace:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()

        lr_updater = parameter.PerEpochLossUpdater(
            learning_rate_v, steps_per_epoch=1000)

        scaffold = tf.train.Scaffold(
            local_init_op=tf.group(tf.local_variables_initializer(),
                                   iterator.initializer),
            init_feed_dict={learning_rate: learning_rate_v()})
        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=train_dir,
                hooks=[
                    tf.train.StopAtStepHook(num_steps=max_steps),
                    tf.train.NanTensorHook(measure),
                    _LoggerHook()
                ],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(
                    log_device_placement=log_device_placement)) as sess:

            while not sess.should_stop():

                _, _, _, _, loss_v = sess.run(
                    [
                        d_training, g_training, update_gain,
                        update_global_step, g_loss
                    ],
                    feed_dict={learning_rate: learning_rate_v()},
                    options=run_options,
                    run_metadata=run_metadata)

                # lr_updater(loss_v)

    return loss_v


def bayesian_optimize(args, options=None):
    if options is None:
        local_options = {'count': 0}
    else:
        local_options = options

    def f(x):
        nonlocal local_options
        print(x)
        evaluation = run_began(
            style_encoder_graph=args.style_encoder_graph,
            dataset_dir=args.dataset_dir,
            train_dir=str(
                Path(args.train_dir, "log_{}".format(local_options['count']))),
            log_device_placement=args.log_device_placement,
            full_trace=args.full_trace,
            learning_rate=args.learning_rate,
            beta1=float(x[:, 0]),
            balance=float(x[:, 1]),
            gain_balance=float(x[:, 2]),
            batch_size=int(x[:, 3]),
            max_steps=30000)

        local_options['count'] += 1
        return evaluation

    return f, local_options


if __name__ == '__main__':
    argparser = create_arg_parser()
    args = argparser.parse_args()

    bounds = [
        {
            'name': 'beta1',
            'type': 'continuous',
            'domain': (0.5, 0.9)
        },
        {
            'name': 'balance',
            'type': 'continuous',
            'domain': (0.25, 0.75)
        },
        {
            'name': 'gain_balance',
            'type': 'continuous',
            'domain': (0.001, 0.1)
        },
        {
            'name': 'batch_size',
            'type': 'discrete',
            'domain': (2, 3, 4, 5)
        }
    ]

    stored_bayesian_param_file = Path(args.train_dir, "bayesian.npz")
    stored_option = Path(args.train_dir, "options.pickle")

    options = None
    X = None
    Y = None
    if stored_bayesian_param_file.exists():
        npzfile = np.load(str(stored_bayesian_param_file))
        X = npzfile['arr_0']
        Y = npzfile['arr_1']

    if stored_option.exists():
        with open(str(stored_option), "rb") as f:
            options = pickle.load(f)

    f, options = bayesian_optimize(args, options)
    opt_gan = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, X=X, Y=Y)
    try:
        opt_gan.run_optimization(max_iter=50)
        print("Optimized parameters: {}".format(opt_gan.x_opt))
        print("Optimized loss: {}".format(opt_gan.fx_opt))
    finally:
        print("Save param files")
        np.savez_compressed(
            str(stored_bayesian_param_file), opt_gan.x_opt, opt_gan.fx_opt)
        with open(str(stored_option), "wb") as f:
            pickle.dump(options, f)
