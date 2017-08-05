# coding: utf-8

import tensorflow as tf
from tflib import operations as op


class Generator(object):
    def __init__(self, channels):
        self.bnc0 = op.BatchNormalization(name='bnc0')
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')
        self.bnc5 = op.BatchNormalization(name='bnc5')
        self.bnc6 = op.BatchNormalization(name='bnc6')
        self.bnc7 = op.BatchNormalization(name='bnc7')
        self.bnc8 = op.BatchNormalization(name='bnc8')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd5 = op.BatchNormalization(name='bnd5')
        self.bnd6 = op.BatchNormalization(name='bnd6')
        self.bnd7 = op.BatchNormalization(name='bnd7')
        self.bnd8 = op.BatchNormalization(name='bnd8')

        self.conv0 = op.Encoder(channels, 32, 3, 3, strides=[1, 1, 1, 1], name='encoder0')
        self.conv1 = op.Encoder(32, 64, 4, 4, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = op.Encoder(64, 64, 3, 3, strides=[1, 1, 1, 1], name='encoder2')
        self.conv3 = op.Encoder(64, 128, 4, 4, strides=[1, 2, 2, 1], name='encoder3')
        self.conv4 = op.Encoder(128, 128, 3, 3, strides=[1, 1, 1, 1], name='encoder4')
        self.conv5 = op.Encoder(128, 256, 4, 4, strides=[1, 2, 2, 1], name='encoder5')
        self.conv6 = op.Encoder(256, 256, 3, 3, strides=[1, 1, 1, 1], name='encoder6')
        self.conv7 = op.Encoder(256, 512, 4, 4, strides=[1, 2, 2, 1], name='encoder7')
        self.conv8 = op.Encoder(512, 512, 3, 3, strides=[1, 1, 1, 1], name='encoder8')

        self.linear = op.LinearEncoder(128)

        self.deconv8 = op.PixelShuffler(op.Encoder(1152, 1024, 3, 3, name='decoder8'), 256, 2)
        self.deconv7 = op.Encoder(256, 256, 3, 3, name='decoder7')
        self.deconv6 = op.PixelShuffler(op.Encoder(512, 512, 3, 3, name='decoder6'), 128, 2)
        self.deconv5 = op.Encoder(128, 128, 3, 3, name='decoder5')
        self.deconv4 = op.PixelShuffler(op.Encoder(256, 256, 3, 3, name='decoder4'), 64, 2)
        self.deconv3 = op.Encoder(64, 64, 3, 3, name='decoder3')
        self.deconv2 = op.PixelShuffler(op.Encoder(128, 128, 3, 3, name='decoder2'), 32, 2)
        self.deconv1 = op.Encoder(32, 32, 3, 3, name='decoder1')
        self.deconv0 = op.Encoder(64, 3, 3, 3, name='decoder0')


def generator(image, tag):
    """Make construction layer.
    """
    channels = image.shape.as_list()[3]
    gen = Generator(channels)

    relu = tf.nn.relu

    conv0 = relu(gen.bnc0(gen.conv0(image)))
    conv1 = relu(gen.bnc1(gen.conv1(conv0)))
    conv2 = relu(gen.bnc2(gen.conv2(conv1)))
    conv3 = relu(gen.bnc3(gen.conv3(conv2)))
    conv4 = relu(gen.bnc4(gen.conv4(conv3)))
    conv5 = relu(gen.bnc5(gen.conv5(conv4)))
    conv6 = relu(gen.bnc6(gen.conv6(conv5)))
    conv7 = relu(gen.bnc7(gen.conv7(conv6)))
    conv8 = relu(gen.bnc8(gen.conv8(conv7)))

    linear = gen.linear(tag, tag.shape.as_list()[1])
    shape = conv8.shape.as_list()
    replicated = tf.tile(tf.reshape(linear, [-1, 1, 1, 128]), [1, shape[1], shape[2], 1])

    deconv8 = relu(gen.bnd8(gen.deconv8(tf.concat([conv8, conv7, replicated], 3))))
    deconv7 = relu(gen.bnd7(gen.deconv7(deconv8)))
    deconv6 = relu(gen.bnd6(gen.deconv6(tf.concat([deconv7, conv6], 3))))
    deconv5 = relu(gen.bnd5(gen.deconv5(deconv6)))
    deconv4 = relu(gen.bnd4(gen.deconv4(tf.concat([deconv5, conv4], 3))))
    deconv3 = relu(gen.bnd3(gen.deconv3(deconv4)))
    deconv2 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv3, conv2], 3))))
    deconv1 = relu(gen.bnd1(gen.deconv1(deconv2)))
    deconv0 = tf.nn.tanh(gen.deconv0(tf.concat([deconv1, conv0], 3)))

    return deconv0


class Critic(object):
    """Define critic"""

    def __init__(self, in_ch):

        self.ln1 = op.LayerNormalization(name="layer_norm1")
        self.ln3 = op.LayerNormalization(name="layer_norm3")
        self.ln5 = op.LayerNormalization(name="layer_norm5")
        self.ln7 = op.LayerNormalization(name="layer_norm7")

        self.conv1 = op.Encoder(in_ch, 64, 3, 3, name='encoder1', strides=[1, 2, 2, 1])
        self.conv3 = op.Encoder(64, 128, 3, 3, name='encoder3', strides=[1, 2, 2, 1])
        self.conv5 = op.Encoder(128, 256, 3, 3, name='encoder5', strides=[1, 2, 2, 1])
        self.conv7 = op.Encoder(256, 512, 3, 3, name='encoder7', strides=[1, 2, 2, 1])

        self.fully_connect = op.Dense("fully_connect1")
        self.linear = op.LinearEncoder(128)

    def __call__(self, tensor, tag):
        net = tf.nn.relu(self.ln1(self.conv1(tensor)))
        net = tf.nn.relu(self.ln3(self.conv3(net)))
        net = tf.nn.relu(self.ln5(self.conv5(net)))
        net = tf.nn.relu(self.ln7(self.conv7(net)))

        shape = net.shape.as_list()
        linear = self.linear(tag, tag.shape.as_list()[1])
        replicated = tf.tile(tf.reshape(linear, [-1, 1, 1, 128]), [1, shape[1], shape[2], 1])

        net = self.fully_connect(tf.concat([net, replicated], 3), 1)

        return tf.reshape(net, [-1])


def critic(base, originals, tags):
    """make critic network"""

    chan = originals.shape.as_list()[3] + base.shape.as_list()[3]
    C = Critic(chan)
    logit = C(tf.concat([base, originals], 3), tags)

    return logit


def gradient_penalty(diff, interpolates, lambda_):
    """compute gradient penalty via https://arxiv.org/pdf/1704.00028.pdf
    """

    # compute gradient ∇xtDw(tx)
    gradients = tf.gradients(diff, [interpolates])[0]
    # compute l2-norm of gradients
    delta = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    penalty = tf.reduce_mean(tf.square(delta - 1))

    return lambda_ * penalty


def c_loss(real, fake):
    # maximize 1/m(E(f(x)) - E(f(g(x))))
    # tensorflow's optimizers can only be minimizing gradient, so
    # this loss applied unary negative operator.

    loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    return loss


def g_loss(logit):
    # minimize -1/mΣEf(g(x))
    loss = -tf.reduce_mean(logit)

    return loss


def l1_loss(original, gen):
    l1_distance = tf.reduce_mean(tf.abs(original - gen)) * 100

    return l1_distance


class AdamTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(loss, global_step=self.global_step, var_list=var_list,
                                        colocate_gradients_with_ops=True)

        return train_step
