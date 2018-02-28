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
        self.bnd2_1 = op.BatchNormalization(name='bnd2_1')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')
        self.bnd4_1 = op.BatchNormalization(name='bnd4_1')
        self.bnd5 = op.BatchNormalization(name='bnd5')
        self.bnd6 = op.BatchNormalization(name='bnd6')
        self.bnd6_1 = op.BatchNormalization(name='bnd6_1')
        self.bnd7 = op.BatchNormalization(name='bnd7')
        self.bnd8 = op.BatchNormalization(name='bnd8')

        self.conv0 = op.Encoder(channels, 32, 3, 3, strides=[1, 1, 1, 1], name='encoder0')
        self.conv1 = op.Encoder(32, 32, 4, 4, strides=[1, 1, 1, 1], name='encoder1')
        self.conv2 = op.Encoder(32, 64, 4, 4, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = op.Encoder(64, 64, 3, 3, strides=[1, 1, 1, 1], name='encoder3')
        self.conv4 = op.Encoder(64, 128, 4, 4, strides=[1, 2, 2, 1], name='encoder4')
        self.conv5 = op.Encoder(128, 128, 3, 3, strides=[1, 1, 1, 1], name='encoder5')
        self.conv6 = op.Encoder(128, 256, 4, 4, strides=[1, 2, 2, 1], name='encoder6')
        self.conv7 = op.Encoder(256, 256, 3, 3, strides=[1, 1, 1, 1], name='encoder7')
        self.conv8 = op.Encoder(256, 1024, 4, 4, strides=[1, 2, 2, 1], name='encoder8')

        self.linear = op.LinearEncoder(1024)

        self.deconv8 = op.PixelShuffler(op.Encoder(2048, 2048, 3, 3, name='decoder8'), 512, 2)
        self.deconv7 = op.Encoder(512 + 256, 512, 3, 3, name='decoder7')
        self.deconv6_1 = op.Encoder(512 + 256, 512, 3, 3, name='decoder6_1')
        self.deconv6 = op.PixelShuffler(op.Encoder(512 + 256, 512, 3, 3, name='decoder6'), 128, 2)
        self.deconv5 = op.Encoder(128 + 128, 128, 3, 3, name='decoder5')
        self.deconv4_1 = op.Encoder(128 + 128, 128, 3, 3, name='decoder4_1')
        self.deconv4 = op.PixelShuffler(op.Encoder(128 + 128, 256, 3, 3, name='decoder4'), 64, 2)
        self.deconv3 = op.Encoder(64 + 64, 64, 3, 3, name='decoder3')
        self.deconv2_1 = op.Encoder(64 + 64, 64, 3, 3, name='decoder2_1')
        self.deconv2 = op.PixelShuffler(op.Encoder(64 + 64, 128, 3, 3, name='decoder2'), 32, 2)
        self.deconv1 = op.Encoder(32 + 32, 32, 3, 3, name='decoder1')
        self.deconv0 = op.Encoder(32 + 32, 3, 3, 3, name='decoder0')


def generator(image, style):
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
    _, style_channel = style.shape.as_list()
    replicated = tf.tile(tf.reshape(style, [-1, 1, 1, style_channel]),
                         [1, shape[1], shape[2], 1])

    deconv8 = relu(gen.bnd8(gen.deconv8(tf.concat([conv8, replicated], 3))))
    deconv7 = relu(gen.bnd7(gen.deconv7(tf.concat([deconv8, conv7], 3))))
    deconv6 = relu(gen.bnd6(gen.deconv6(tf.concat([deconv7, conv6], 3))))
    deconv5 = relu(gen.bnd5(gen.deconv5(tf.concat([deconv6, conv5], 3))))
    deconv4 = relu(gen.bnd4(gen.deconv4(tf.concat([deconv5, conv4], 3))))
    deconv3 = relu(gen.bnd3(gen.deconv3(tf.concat([deconv4, conv3], 3))))
    deconv2 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv3, conv2], 3))))
    deconv1 = relu(gen.bnd1(gen.deconv1(tf.concat([deconv2, conv1], 3))))
    deconv0 = tf.nn.tanh(gen.deconv0(tf.concat([deconv1, conv0], 3)))

    return conv8, deconv7, deconv0

class GuideDecoder1(object):
    def __init__(self):
        self.bnd1 = op.BatchNormalization(name='guide1_bnd1')
        self.bnd2 = op.BatchNormalization(name='guide1_bnd2')
        self.bnd3 = op.BatchNormalization(name='guide1_bnd3')
        self.bnd4 = op.BatchNormalization(name='guide1_bnd4')

        self.deconv4 = op.PixelShuffler(None, 256, 2)
        self.deconv3 = op.PixelShuffler(op.Encoder(256, 512, 3, 3, name='guide1_decoder3'), 128, 2)
        self.deconv2 = op.PixelShuffler(op.Encoder(128, 256, 3, 3, name='guide1_decoder2'), 64, 2)
        self.deconv1 = op.PixelShuffler(op.Encoder(64, 128, 3, 3, name='guide1_decoder1'), 32, 2)
        self.deconv0 = op.Encoder(32, 3, 3, 3, name="guide1_decoder0")

def guide_decoder1(last_conv):
    dec = GuideDecoder1()

    relu = tf.nn.relu
    net = relu(dec.bnd4(dec.deconv4(last_conv)))
    net = relu(dec.bnd3(dec.deconv3(net)))
    net = relu(dec.bnd2(dec.deconv2(net)))
    net = relu(dec.bnd1(dec.deconv1(net)))

    return tf.nn.tanh(dec.deconv0(net))

class GuideDecoder2(object):
    def __init__(self):
        self.bnd1 = op.BatchNormalization(name='guide2_bnd1')
        self.bnd2 = op.BatchNormalization(name='guide2_bnd2')
        self.bnd3 = op.BatchNormalization(name='guide2_bnd3')

        self.deconv3 = op.PixelShuffler(op.Encoder(512, 512, 3, 3, name='guide2_decoder3'), 128, 2)
        self.deconv2 = op.PixelShuffler(op.Encoder(128, 256, 3, 3, name='guide2_decoder2'), 64, 2)
        self.deconv1 = op.PixelShuffler(op.Encoder(64, 128, 3, 3, name='guide2_decoder1'), 32, 2)
        self.deconv0 = op.Encoder(32, 3, 3, 3, name="guide2_decoder0")

def guide_decoder2(last_conv):
    dec = GuideDecoder2()

    relu = tf.nn.relu
    net = relu(dec.bnd3(dec.deconv3(last_conv)))
    net = relu(dec.bnd2(dec.deconv2(net)))
    net = relu(dec.bnd1(dec.deconv1(net)))

    return tf.nn.tanh(dec.deconv0(net))

class Discriminator(object):
    def __init__(self, channels):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc1_f = op.BatchNormalization(name='bnc1_f')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc2_f = op.BatchNormalization(name='bnc2_f')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc3_f = op.BatchNormalization(name='bnc3_f')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd1_f = op.BatchNormalization(name='bnd1_f')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd2_f = op.BatchNormalization(name='bnd2_f')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd3_f = op.BatchNormalization(name='bnd3_f')

        self.conv1 = op.Encoder(channels, 64, 3, 3, strides=[1, 2, 2, 1], name='encoder1')
        self.conv1_f1 = op.Encoder(64, 64, 3, 3, name="encoder1_f1")
        self.conv2 = op.Encoder(64, 128, 3, 3, strides=[1, 2, 2, 1], name='encoder2')
        self.conv2_f1 = op.Encoder(128, 128, 3, 3, name="encoder2_f1")
        self.conv3 = op.Encoder(128, 256, 3, 3, strides=[1, 2, 2, 1], name='encoder3')
        self.conv3_f1 = op.Encoder(256, 256, 3, 3, name="encoder3_f1")

        self.deconv3 = op.PixelShuffler(op.Encoder(256, 512, 3, 3, name='decoder3'), 128, 2)
        self.deconv3_f1 = op.Encoder(128, 128, 3, 3, name="decoder3_f1")
        self.deconv2 = op.PixelShuffler(op.Encoder(256, 256, 3, 3, name='decoder2'), 64, 2)
        self.deconv2_f1 = op.Encoder(64, 64, 3, 3, name="decoder2_f1")
        self.deconv1 = op.PixelShuffler(op.Encoder(128, 128, 3, 3, name='decoder1'), 32, 2)
        self.deconv1_f1 = op.Encoder(32, 32, 3, 3, name="decoder1_f1")
        self.deconv0 = op.Encoder(32, 3, 3, 3, name="decoder0")

        self.fully_connect = op.Dense('fully_connect')
        self.fully_unconnect = op.Dense('fully_unconnect')


def discriminator(img):
    """Make construction layer.
    """

    channels = img.shape.as_list()[3]
    D = Discriminator(channels)

    relu = tf.nn.relu

    net = img
    net = relu(D.bnc1(D.conv1(net)))
    net = relu(D.bnc1_f(D.conv1_f1(net)))
    net = relu(D.bnc2(D.conv2(net)))
    net = relu(D.bnc2_f(D.conv2_f1(net)))
    net = relu(D.bnc3(D.conv3(net)))
    net = relu(D.bnc3_f(D.conv3_f1(net)))

    _, h, w, c = net.get_shape().as_list()
    net = D.fully_connect(net, 128)
    net = D.fully_unconnect(net, h * w * c)
    net = tf.reshape(net, [-1, h, w, c])

    net = deconv3 = relu(D.bnd3(D.deconv3(net)))
    net = relu(D.bnd3_f(D.deconv3_f1(net)))
    net = deconv2 = relu(D.bnd2(D.deconv2(tf.concat([net, deconv3], 3))))
    net = relu(D.bnd2_f(D.deconv2_f1(net)))
    net = relu(D.bnd1(D.deconv1(tf.concat([net, deconv2], 3))))
    net = relu(D.bnd1_f(D.deconv1_f1(net)))
    net = tf.nn.tanh(D.deconv0(net))

    return net


def d_loss(real, real_pred, gen, gen_pred, gain):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    real_loss = tf.reduce_mean(tf.square(real - real_pred), [1,2,3])
    gen_loss = tf.reduce_mean(tf.square(gen - gen_pred), [1,2,3])

    loss = tf.reduce_mean(real_loss - gen_loss * gain)
    return loss


def g_loss(gen, gen_pred, original, guide1, guide2, gray_original, alpha=0.3, beta=0.7):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    penalty = tf.reduce_mean(tf.abs(gen - original), [1,2,3])
    g_loss = tf.reduce_mean(tf.abs(gen - gen_pred), [1,2,3])
    g1_loss = alpha * tf.reduce_mean(tf.abs(gray_original - guide1), [1,2,3])
    g2_loss = beta * tf.reduce_mean(tf.abs(original - guide2), [1,2,3])

    loss = tf.reduce_mean(g_loss + g1_loss + g2_loss + penalty)
    return loss


def balanced_d_loss(real, real_pred, gen, gen_pred, balance):
    """Calculate balanced D loss.
    """
    real_loss = tf.reduce_mean(tf.square(real - real_pred), [1,2,3])
    gen_loss = tf.reduce_mean(tf.square(gen - gen_pred), [1,2,3])

    loss = tf.reduce_mean(balance * real_loss - gen_loss)
    return loss


def global_measure(real, real_pred, balanced_loss):
    # global convergence is calculated by |D(v) - D(G(v))|
    d_loss = tf.reduce_mean(tf.square(real - real_pred), [1,2,3])
    measure = tf.reduce_mean(d_loss + tf.abs(balanced_loss))

    return measure


class Trainer(object):
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
