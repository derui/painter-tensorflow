# coding: utf-8

import tensorflow as tf
from . import operations as op


# Define weight variable
def weight_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


# Define bias variable
def bias_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_h, patch_w, strides=[1, 1, 1, 1], name='encoder'):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.in_ch, self.out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(tensor, weight, strides=self.strides, padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_h, patch_w, batch_size, strides=[1, 1, 1, 1], name='decoder'):
        self.batch_size = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, output_shape):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.out_ch, self.in_ch], name='{}_weight'.format(self.name))

        bias = bias_variable([self.out_ch], name="{}_bias".format(self.name))

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight, [self.batch_size, output_shape[0], output_shape[1], self.out_ch],
            strides=self.strides,
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Generator(object):
    def __init__(self, batch_size):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')
        self.bnc5 = op.BatchNormalization(name='bnc5')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd5 = op.BatchNormalization(name='bnd5')

        self.conv1 = Encoder(1, 64, 4, 4, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = Encoder(64, 128, 4, 4, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = Encoder(128, 256, 4, 4, strides=[1, 2, 2, 1], name='encoder3')
        self.conv4 = Encoder(256, 512, 4, 4, strides=[1, 2, 2, 1], name='encoder4')
        self.conv5 = Encoder(512, 1024, 4, 4, strides=[1, 1, 1, 1], name='encoder5')

        # self.deconv1 = Decoder(1024, 512, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder1')
        # self.deconv2 = Decoder(1024, 256, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder2')
        # self.deconv3 = Decoder(512, 128, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder3')
        # self.deconv4 = Decoder(256, 64, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder4')
        # self.deconv5 = Decoder(128, 3, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder5')

        self.deconv1 = Encoder(1024, 512, 4, 4, name='decoder1')
        self.deconv2 = op.PixelShuffler(Encoder(1024, 1024, 4, 4, name='decoder2'), 256, 2)
        self.deconv3 = op.PixelShuffler(Encoder(512, 512, 4, 4, name='decoder3'), 128, 2)
        self.deconv4 = op.PixelShuffler(Encoder(256, 256, 4, 4, name='decoder4'), 64, 2)
        self.deconv5 = op.PixelShuffler(Encoder(128, 128, 4, 4, name='decoder5'), 32, 2)
        self.flatconv = Encoder(32, 3, 4, 4, name='flatconv')


def generator(image):
    """Make construction layer.
    """
    batch_size, height, width, channels = image.shape.as_list()
    gen = Generator(batch_size)

    relu = tf.nn.relu

    def lrelu(x):
        return tf.maximum(0.2 * x, x)

    conv1 = relu(gen.bnc1(gen.conv1(image)))
    conv2 = relu(gen.bnc2(gen.conv2(conv1)))
    conv3 = relu(gen.bnc3(gen.conv3(conv2)))
    conv4 = relu(gen.bnc4(gen.conv4(conv3)))
    conv5 = relu(gen.bnc5(gen.conv5(conv4)))

    deconv1 = relu(gen.bnd1(gen.deconv1(conv5)))
    deconv2 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv1, conv4], 3))))
    deconv3 = relu(gen.bnd3(gen.deconv3(tf.concat([deconv2, conv3], 3))))
    deconv4 = relu(gen.bnd4(gen.deconv4(tf.concat([deconv3, conv2], 3))))
    deconv5 = relu(gen.bnd5(gen.deconv5(tf.concat([deconv4, conv1], 3))))
    deconv5 = tf.nn.tanh(gen.flatconv(deconv5))

    return deconv5


class Dense(object):
    """The dense layer """

    def __init__(self, name='dense'):
        self.name = name

    def __call__(self, tensor, out_ch):
        _, w, h, c = tensor.shape.as_list()
        in_ch = w*h*c
        weight = weight_variable(
            [in_ch, out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.bias_add(tf.matmul(tf.reshape(tensor, [-1, in_ch]), weight), bias)

        return conv


class Critic(object):
    """Define critic"""

    def __init__(self, in_ch):

        self.conv1 = Encoder(in_ch, 64, 3, 3, name='encoder1', strides=[1, 2, 2, 1])
        self.conv2 = Encoder(64, 128, 3, 3, name='encoder2', strides=[1, 2, 2, 1])
        self.conv3 = Encoder(128, 256, 3, 3, name='encoder3', strides=[1, 2, 2, 1])
        self.conv4 = Encoder(256, 512, 3, 3, name='encoder4', strides=[1, 2, 2, 1])

        self.fully_connect = Dense("fully_connect1")

    def __call__(self, tensor):
        bs, width, height, _ = tensor.shape.as_list()

        def lrelu(x):
            return tf.maximum(0.2 * x, x)

        net = lrelu(self.conv1(tensor))
        net = lrelu(self.conv2(net))
        net = lrelu(self.conv3(net))
        net = lrelu(self.conv4(net))

        net = self.fully_connect(net, 1)

        return tf.reshape(net, [-1])


def critic(bases, originals):
    """make critic network"""

    chan = bases.shape.as_list()[3] + originals.shape.as_list()[3]
    C = Critic(chan)
    logit = C(tf.concat([originals, bases], 3))

    return logit


def gradient_penalty(diff, interpolates, lambda_):
    """compute gradient penalty via https://arxiv.org/pdf/1704.00028.pdf
    """

    # compute gradient ∇xtDw(tx)
    gradients = tf.gradients(diff, [interpolates])[0]
    # compute l2-norm of gradients
    delta = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    penalty = tf.reduce_mean(tf.square(delta - 1))

    return lambda_ * penalty


def c_loss(real, fake, gradient_penalty):
    with tf.name_scope('c_loss'):
        # maximize 1/m(E(f(x)) - E(f(g(x))))
        # tensorflow's optimizers can only be minimizing gradient, so
        # this loss applied unary negative operator.

        loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        tf.summary.scalar('grad', gradient_penalty)
        tf.summary.scalar('c_entropy', loss + gradient_penalty)
    return loss + gradient_penalty


def g_loss(logit):
    with tf.name_scope('g_loss'):
        # minimize -1/mΣEf(g(x))
        cross_entropy = -tf.reduce_mean(logit)
        tf.summary.scalar('g_entropy', cross_entropy)

    return cross_entropy


def l1_loss(original, gen):
    with tf.name_scope('l1_loss'):

        l1_distance = tf.reduce_mean(tf.reduce_sum(tf.abs(original - gen), axis=[1,2,3])) * 0.1

        tf.summary.scalar('distance', l1_distance)

    return l1_distance


class Trainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, var_list):
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_step = optimizer.minimize(loss, global_step=self.global_step, var_list=var_list)

        return train_step


class AdamTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(loss, global_step=self.global_step, var_list=var_list)

        return train_step
