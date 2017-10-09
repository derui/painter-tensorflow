# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class EmbeddingNet(object):
    def __init__(self, channels):
        self.bnc1 = op.BatchNormalization(name='embedding/bnc1')
        self.bnc2 = op.BatchNormalization(name='embedding/bnc2')

        self.conv1 = op.Encoder(channels, 256, 3, 3, name='embedding/encoder0')
        self.conv2 = op.Encoder(256, 256, 3, 3, name='embedding/encoder1')

    def __call__(self, tensor):

        in_tensor = tf.nn.relu(self.bnc1(self.conv1(tensor)))
        net = tf.nn.relu(self.bnc2(self.conv2(in_tensor)))

        return in_tensor, net


class InferenceNet(object):
    def __init__(self, channels, recursives=5):

        self.encoder = op.Encoder(channels, channels, 3, 3, name="encoder", initializer=tf.constant_initializer(0.0))

        self.recursives = recursives
        self.bn = op.BatchNormalization(name="normalization")

    def __call__(self, tensor):
        outputs = (self.recursives + 1) * [None]
        outputs[0] = tensor

        with tf.variable_scope("inference") as scope:
            do_reuse = True

            for i in range(self.recursives):
                outputs[i + 1] = tf.nn.relu((self.encoder(outputs[i])))

                if do_reuse:
                    do_reuse = False
                    scope.reuse_variables()

        return outputs[1:]


class Upsampler(object):
    def __init__(self, inference_channels, original_channels, recursives=5):
        self.bnc1 = op.BatchNormalization(name='upsampler/bnc1')
        self.bnc2 = op.BatchNormalization(name='upsampler/bnc2')

        self.conv1 = op.PixelShuffler(op.Encoder(inference_channels * 2, 256, 3, 3, name='upsampler/encoder1'), 64, 2)
        self.conv2 = op.PixelShuffler(op.Encoder(64, 128, 3, 3, name="upsampler/encoder2"), 32, 2)
        self.conv3 = op.Encoder(32, 3, 3, 3, name="upsampler/encoder3")

        self.recursives = recursives

    def __call__(self, original, tensors):

        result = None
        outputs = self.recursives * [None]
        reuse = True
        weight_sum = None
        self.weights = op.weight_variable([self.recursives], name="decay", initializer=tf.constant_initializer(0.2))

        with tf.variable_scope("upsampler") as scope:
            weight_sum = tf.reduce_sum(self.weights)

            for i in range(self.recursives):
                outputs[i] = tf.nn.relu(self.bnc1(self.conv1(tf.concat([original, tensors[i]], 3))))
                outputs[i] = tf.nn.relu(self.bnc2(self.conv2(outputs[i])))
                outputs[i] = tf.nn.tanh(self.conv3(outputs[i]))

                if result is None:
                    result = tf.multiply(outputs[i], self.weights[i]) / weight_sum
                else:
                    result += tf.multiply(outputs[i], self.weights[i]) / weight_sum

                if reuse:
                    reuse = False
                    scope.reuse_variables()

                tf.summary.image('output{}'.format(i), outputs[i])

        tf.summary.histogram('weight', self.weights)

        return tf.nn.tanh(result), outputs


def upsample(small, recursives):
    """Make construction layer.
    """
    channels = small.shape.as_list()[3]
    embedded_net = EmbeddingNet(channels)
    in_tensor, net = embedded_net(small)

    embedding_shape = net.shape.as_list()
    inferences = InferenceNet(embedding_shape[3], recursives=recursives)

    nets = inferences(net)

    upsampler = Upsampler(embedding_shape[3], channels, recursives)

    return upsampler(in_tensor, nets)


def upsample_loss(original, intermediates, gen, alpha, beta_loss, scope):
    losses = []
    for i in range(len(intermediates)):
        losses.append(tf.reduce_mean(tf.square(original - intermediates[i]), axis=[1, 2, 3]))

    l1 = tf.reduce_mean(losses)

    l2 = tf.reduce_mean(tf.square(original - gen))

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    l3 = None
    for v in var_list:
        if '_weight' not in v.name and '_bias' not in v.name:
            continue

        if l3 is None:
            l3 = tf.nn.l2_loss(v)
        else:
            l3 += tf.nn.l2_loss(v)

    tf.summary.scalar("loss_l1", l1)
    tf.summary.scalar("loss_l2", l2)
    tf.summary.scalar("loss_l3", l3 * beta_loss)

    return (alpha * l1) + ((1 - alpha) * l2) + (beta_loss * l3)


class AdamTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list=None):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(
            loss, global_step=self.global_step, var_list=var_list, colocate_gradients_with_ops=True)

        return train_step


class RMSPropTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, var_list=None):
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_step = optimizer.minimize(
            loss, global_step=self.global_step, var_list=var_list, colocate_gradients_with_ops=True)

        return train_step
