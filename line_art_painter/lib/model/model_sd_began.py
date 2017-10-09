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

        self.deconv8 = op.PixelShuffler(op.Encoder(1152, 1024, 3, 3, name='decoder8'), 256, 2)
        self.deconv7 = op.Encoder(256, 256, 3, 3, name='decoder7')
        self.deconv6 = op.PixelShuffler(op.Encoder(512, 512, 3, 3, name='decoder6'), 128, 2)
        self.deconv5 = op.Encoder(128, 128, 3, 3, name='decoder5')
        self.deconv4 = op.PixelShuffler(op.Encoder(256, 256, 3, 3, name='decoder4'), 64, 2)
        self.deconv3 = op.Encoder(64, 64, 3, 3, name='decoder3')
        self.deconv2 = op.PixelShuffler(op.Encoder(128, 128, 3, 3, name='decoder2'), 32, 2)
        self.deconv1 = op.Encoder(32, 32, 3, 3, name='decoder1')
        self.deconv0 = op.Encoder(64, 3, 3, 3, name='decoder0')


def generator(image, noise):
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

    shape = conv8.shape.as_list()
    replicated = tf.tile(tf.reshape(noise, [-1, 1, 1, 128]), [1, shape[1], shape[2], 1])

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


class Discriminator(object):
    """Define discriminator"""

    def __init__(self, in_ch):

        self.bnc1 = op.BatchNormalization(name="bnc1")
        self.bnc2 = op.BatchNormalization(name="bnc2")
        self.bnc3 = op.BatchNormalization(name="bnc3")
        self.bnc4 = op.BatchNormalization(name="bnc4")

        self.conv1 = op.Encoder(in_ch, 64, 3, 3, name='encoder1', strides=[1, 2, 2, 1])
        self.conv2 = op.Encoder(64, 128, 3, 3, name='encoder2', strides=[1, 2, 2, 1])
        self.conv3 = op.Encoder(128, 256, 3, 3, name='encoder3', strides=[1, 2, 2, 1])
        self.conv4 = op.Encoder(256, 512, 3, 3, name='encoder4', strides=[1, 2, 2, 1])

        self.dense = op.Dense("dense")

    def __call__(self, tensor):
        net = tf.nn.relu(self.bnc1(self.conv1(tensor)))
        net = tf.nn.relu(self.bnc2(self.conv2(net)))
        net = tf.nn.relu(self.bnc3(self.conv3(net)))
        net = tf.nn.relu(self.bnc4(self.conv4(net)))

        net = self.dense(net, 128)

        return net


def discriminator(x, layer):
    """make critic network"""

    chan = x.shape.as_list()[3]
    D = Discriminator(chan)
    logit = D(x)

    with tf.variable_scope('dis_gen'):
        return generator(layer, logit)


def d_loss(real, real_pred, gen, gen_pred, gain):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    real_loss = tf.reduce_mean(tf.abs(real - real_pred), axis=[1, 2, 3])
    gen_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1, 2, 3])

    loss = tf.reduce_mean(real_loss - gen_loss * gain)
    return loss


def g_loss(gen, gen_pred, original):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    original_loss = tf.reduce_mean(tf.abs(gen - original), axis=[1, 2, 3])
    g_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1, 2, 3])

    loss = tf.reduce_mean(g_loss + original_loss)
    return loss


def balanced_d_loss(real, real_pred, gen, gen_pred, balance):
    """Calculate balanced D loss.
    """
    real_loss = tf.reduce_mean(tf.abs(real - real_pred), axis=[1, 2, 3])
    gen_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1, 2, 3])

    loss = tf.reduce_mean(balance * real_loss - gen_loss)
    return loss


class AdamTrainer(object):
    """
    Wrap up training function in this model.

    This class should create instance per training.
    """

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __call__(self, loss, learning_rate, beta1, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        train_step = optimizer.minimize(
            loss, global_step=self.global_step, var_list=var_list, colocate_gradients_with_ops=True)

        return train_step
