# coding: utf-8

import tensorflow as tf
from tflib import operations as op


def weight_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


def bias_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_w, patch_h, strides=[1, 1, 1, 1], name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_w, self.patch_h, self.in_ch, self.out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(tensor, weight, strides=self.strides, padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Decoder(object):
    """The decoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self, in_ch, out_ch, patch_w, patch_h, strides=[1, 1, 1, 1], name='encoder'):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, output_shape):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.out_ch, self.in_ch], name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d_transpose(
            tensor,
            weight, [tensor.get_shape().as_list()[0], output_shape[0], output_shape[1], self.out_ch],
            strides=self.strides,
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class AutoEncoder(object):
    """Define autoencoder"""

    def __init__(self):
        self.conv1 = Encoder(3, 48, 5, 5, strides=[1, 2, 2, 1], name='encoder1')
        self.conv1_f1 = Encoder(48, 128, 3, 3, name='encoder1_flat1')
        self.conv1_f2 = Encoder(128, 128, 3, 3, name='encoder1_flat2')
        self.conv2 = Encoder(128, 256, 5, 5, strides=[1, 2, 2, 1], name='encoder2')
        self.conv2_f1 = Encoder(256, 256, 3, 3, name='encoder2_flat1')
        self.conv2_f2 = Encoder(256, 256, 3, 3, name='encoder2_flat2')
        self.conv3 = Encoder(256, 256, 5, 5, strides=[1, 2, 2, 1], name='encoder3')
        self.conv3_f1 = Encoder(256, 512, 3, 3, name='encoder3_flat1')
        self.conv3_f2 = Encoder(512, 1024, 3, 3, name='encoder3_flat2')
        self.conv3_f3 = Encoder(1024, 512, 3, 3, name='encoder3_flat3')
        self.conv3_f4 = Encoder(512, 256, 3, 3, name='encoder3_flat4')

        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc1_f1 = op.BatchNormalization(name='bnc1_flat1')
        self.bnc1_f2 = op.BatchNormalization(name='bnc1_flat2')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc2_f1 = op.BatchNormalization(name='bnc2_flat1')
        self.bnc2_f2 = op.BatchNormalization(name='bnc2_flat2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc3_f1 = op.BatchNormalization(name='bnc3_flat1')
        self.bnc3_f2 = op.BatchNormalization(name='bnc3_flat2')
        self.bnc3_f3 = op.BatchNormalization(name='bnc3_flat3')
        self.bnc3_f4 = op.BatchNormalization(name='bnc3_flat4')

        self.deconv1 = Decoder(256, 256, 4, 4, strides=[1, 2, 2, 1], name='decoder1')
        self.deconv1_f1 = Encoder(256, 128, 3, 3, name='decoder1_flat1')
        self.deconv1_f2 = Encoder(128, 128, 3, 3, name='decoder1_flat2')
        self.deconv2 = Decoder(128, 128, 4, 4, strides=[1, 2, 2, 1], name='decoder2')
        self.deconv2_f1 = Encoder(128, 128, 3, 3, name='decoder2_flat1')
        self.deconv2_f2 = Encoder(128, 48, 3, 3, name='decoder2_flat2')
        self.deconv3 = Decoder(48, 48, 4, 4, strides=[1, 2, 2, 1], name='decoder3')
        self.deconv3_f1 = Decoder(48, 24, 3, 3, name='decoder3_flat1')
        self.deconv3_f2 = Decoder(24, 1, 3, 3, name='decoder3_flat2')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd1_f1 = op.BatchNormalization(name='bnd1_flat1')
        self.bnd1_f2 = op.BatchNormalization(name='bnd1_flat2')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd2_f1 = op.BatchNormalization(name='bnd2_flat1')
        self.bnd2_f2 = op.BatchNormalization(name='bnd2_flat2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd3_f1 = op.BatchNormalization(name='bnd3_flat1')


def autoencoder(images, height, width):
    """make autoencoder network"""

    AE = AutoEncoder()

    def div(v, d):
        return max(1, v // d)

    relu = tf.nn.relu
    net = relu(AE.bnc1(AE.conv1(images, [height, width])))
    net = relu(AE.bnc1_f1(AE.conv1_f1(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnc1_f2(AE.conv1_f2(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnc2(AE.conv2(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnc2_f1(AE.conv2_f1(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnc2_f2(AE.conv2_f2(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnc3(AE.conv3(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnc3_f1(AE.conv3_f1(net, [div(height, 8), div(width, 8)])))
    net = relu(AE.bnc3_f2(AE.conv3_f2(net, [div(height, 8), div(width, 8)])))
    net = relu(AE.bnc3_f3(AE.conv3_f3(net, [div(height, 8), div(width, 8)])))
    net = relu(AE.bnc3_f4(AE.conv3_f4(net, [div(height, 8), div(width, 8)])))
    net = relu(AE.bnd1(AE.deconv1(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnd1_f1(AE.deconv1_f1(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnd1_f2(AE.deconv1_f2(net, [div(height, 4), div(width, 4)])))
    net = relu(AE.bnd2(AE.deconv2(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnd2_f1(AE.deconv2_f1(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnd2_f2(AE.deconv2_f2(net, [div(height, 2), div(width, 2)])))
    net = relu(AE.bnd3(AE.deconv3(net, [height, width])))
    net = relu(AE.bnd3_f1(AE.deconv3_f1(net, [height, width])))

    net = tf.nn.sigmoid(AE.deconv3_f2(net, [height, width]))

    return net


def loss_map(target, bins, alpha, beta):
    """construct loss map from target image"""
    with tf.name_scope('loss_map'):

        patches = tf.extract_image_patches(target, [1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
        minimum = tf.reduce_min(patches, axis=3, keep_dims=True)
        maximum = tf.reduce_max(patches, axis=3, keep_dims=True)

        normalized = (patches - minimum) / (maximum - minimum)
        normalized = tf.floor(patches * bins) / bins

        # make histogram and suppress illegal histogram
        hist = tf.reshape(normalized[:, :, :, 12], target.get_shape())
        hist = tf.where(tf.equal(minimum, maximum), minimum, hist)

        ones = tf.ones_like(target)
        eq = tf.equal(1.0, target)
        hist = tf.where(eq, ones, tf.minimum(1.0, alpha * tf.exp(-hist) + beta))

    return hist


def loss(target, gen, lossmap):

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(lossmap * (gen - target))))

        tf.summary.image('loss_applied', tf.square(lossmap * (gen - target)))

        tf.summary.scalar('loss', loss)
    return loss


def training(loss, learning_rate, global_step, var_list):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_step
