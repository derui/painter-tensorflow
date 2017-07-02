# coding: utf-8

import tensorflow as tf
from . import operations as op


# Define weight variable
def weight_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


# Define bias variable
def bias_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0.0))


class Encoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 patch_h,
                 patch_w,
                 strides=[1, 1, 1, 1],
                 name='encoder'):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, input_shape):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.in_ch, self.out_ch],
            name="{}_weight".format(self.name))
        bias = bias_variable([self.out_ch], name='{}_bias'.format(self.name))
        conv = tf.nn.conv2d(
            tensor, weight, strides=self.strides, padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Decoder(object):
    """The encoder of AutoEncoder. User should give arguments to
    this class that are defined convolutional layer.
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 patch_h,
                 patch_w,
                 batch_size,
                 strides=[1, 1, 1, 1],
                 name='decoder'):
        self.batch_size = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.name = name
        self.strides = strides

    def __call__(self, tensor, output_shape):
        weight = weight_variable(
            [self.patch_h, self.patch_w, self.out_ch, self.in_ch],
            name='{}_weight'.format(self.name))

        bias = bias_variable([self.out_ch], name="{}_bias".format(self.name))

        conv = tf.nn.conv2d_transpose(
            tensor,
            weight,
            [self.batch_size, output_shape[0], output_shape[1], self.out_ch],
            strides=self.strides,
            padding='SAME')
        conv = tf.nn.bias_add(conv, bias)

        return conv


class Generator(object):
    def __init__(self, channels, batch_size):
        self.bnc1 = op.BatchNormalization(name='bnc1')
        self.bnc2 = op.BatchNormalization(name='bnc2')
        self.bnc3 = op.BatchNormalization(name='bnc3')
        self.bnc4 = op.BatchNormalization(name='bnc4')

        self.bnd1 = op.BatchNormalization(name='bnd1')
        self.bnd2 = op.BatchNormalization(name='bnd2')
        self.bnd3 = op.BatchNormalization(name='bnd3')
        self.bnd4 = op.BatchNormalization(name='bnd4')

        self.conv1 = Encoder(channels, 64, 3, 3, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = Encoder(64, 128, 3, 3, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = Encoder(128, 256, 3, 3, strides=[1, 2, 2, 1], name='encoder3')
        self.conv4 = Encoder(256, 512, 3, 3, strides=[1, 2, 2, 1], name='encoder4')

        # self.deconv2 = Decoder(512, 256, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder2')
        # self.deconv3 = Decoder(512, 128, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder3')
        # self.deconv4 = Decoder(256, 64, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder4')
        # self.deconv5 = Decoder(128, 3, 4, 4, batch_size=batch_size, strides=[1, 2, 2, 1], name='decoder5')
        self.deconv1 = Encoder(512, 256, 3, 3, name='decoder1')
        self.deconv2 = Encoder(512, 128, 3, 3, name='decoder2')
        self.deconv3 = Encoder(256, 64, 3, 3, name='decoder3')
        self.deconv4 = Encoder(128, 64, 3, 3, name='decoder4')
        self.deconv5 = Encoder(64, 3, 3, 3, name='decoder5')


def generator(image, height, width, channels, batch_size):
    """Make construction layer.
    """

    gen = Generator(channels, batch_size)

    relu = tf.nn.relu
    tanh = tf.nn.tanh

    conv1 = relu(gen.bnc1(gen.conv1(image, [height, width])))
    conv2 = relu(gen.bnc2(gen.conv2(conv1, [height // 2, width // 2])))
    conv3 = relu(gen.bnc3(gen.conv3(conv2, [height // 4, width // 4])))
    conv4 = relu(gen.bnc4(gen.conv4(conv3, [height // 8, width // 8])))

    deconv1 = relu(gen.bnd1(gen.deconv1(conv4, [height // 16, width // 16])))

    deconv3 = tf.image.resize_images(deconv1, [height // 8, width // 8])
    deconv3 = relu(gen.bnd2(gen.deconv2(tf.concat([deconv3, conv3], 3), [height // 8, width // 8])))

    deconv4 = tf.image.resize_images(deconv3, [height // 4, width // 4])
    deconv4 = relu(gen.bnd3(gen.deconv3(tf.concat([deconv4, conv2], 3), [height // 4, width // 4])))

    deconv5 = tf.image.resize_images(deconv4, [height // 2, width // 2])
    deconv5 = tanh(gen.bnd4(gen.deconv4(tf.concat([deconv5, conv1], 3), [height // 2, width // 2])))

    net = tf.image.resize_images(deconv5, [height, width])
    net = tanh(gen.deconv5(net, [height, width]))

    return net


class Dense(object):
    """The dense layer """

    def __init__(self, name='dense'):
        self.name = name

    def __call__(self, tensor, in_ch, out_ch):
        weight = weight_variable(
            [in_ch, out_ch], name="{}_weight".format(self.name))
        bias = bias_variable([out_ch], name='{}_bias'.format(self.name))
        conv = tf.matmul(tf.reshape(tensor, [-1, in_ch]), weight) + bias

        return conv


class Discriminator(object):
    def __init__(self, channels, batch_size):
        self.conv1 = Encoder(channels, 64, 3, 3, strides=[1, 2, 2, 1], name='encoder1')
        self.conv2 = Encoder(64, 128, 3, 3, strides=[1, 2, 2, 1], name='encoder2')
        self.conv3 = Encoder(128, 256, 3, 3, strides=[1, 2, 2, 1], name='encoder3')

        self.deconv1 = Decoder(256, 128, 3, 3, batch_size=batch_size, strides=[1, 2, 2, 1], name='deconv1')
        self.deconv2 = Decoder(128, 128, 3, 3, batch_size=batch_size, strides=[1, 2, 2, 1], name='deconv2')
        self.deconv3 = Decoder(128, 128, 3, 3, batch_size=batch_size, strides=[1, 2, 2, 1], name='deconv3')
        self.flatconv1 = Encoder(128, 128, 3, 3, strides=[1, 1, 1, 1], name='flatconv1')
        self.flatconv2 = Encoder(256, 128, 3, 3, strides=[1, 1, 1, 1], name='flatconv2')
        self.flatconv3 = Encoder(256, 128, 3, 3, strides=[1, 1, 1, 1], name='flatconv3')
        self.flatconv4 = Encoder(128, 3, 3, 3, strides=[1, 1, 1, 1], name='flatconv4')
        self.upsample1 = Decoder(128, 128, 3, 3, batch_size=batch_size, strides=[1, 2, 2, 1], name='upsample1')
        self.upsample2 = Decoder(128, 128, 3, 3, batch_size=batch_size, strides=[1, 4, 4, 1], name='upsample2')

        self.fully_connect = Dense('fully_connect')
        self.fully_unconnect = Dense('fully_unconnect')


def discriminator(img, height, width, channels, batch_size):
    """Make construction layer.
    """

    D = Discriminator(channels, batch_size)

    relu = tf.nn.relu

    net = relu(D.conv1(img, [height, width]))
    net = relu(D.conv2(net, [height // 2, width // 2]))
    net = relu(D.conv3(net, [height // 4, width // 4]))

    _, h,w,c = net.get_shape().as_list()
    net = D.fully_connect(net, h*w*c, 512)
    net = D.fully_unconnect(net, 512, h*w*c)
    net = tf.reshape(net, [-1, h, w, c])

    l1 = relu(D.deconv1(net, [height // 4, width // 4]))

    net = relu(D.flatconv1(l1, [height // 4, width // 4]))

    net = relu(D.deconv2(net, [height // 2, width // 2]))

    l2 = relu(D.upsample1(l1, [height // 2, width // 2]))

    net = relu(D.flatconv2(tf.concat([net, l2], 3), [height // 2, width // 2]))

    net = relu(D.deconv3(net, [height, width]))
    l3 = relu(D.upsample2(l1, [height, width]))

    net = relu(D.flatconv3(tf.concat([net, l3], 3), [height, width]))
    net = tf.nn.tanh(D.flatconv4(net, [height, width]))

    return net


def d_loss(real, real_pred, gen, gen_pred, gain):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    real_loss = tf.reduce_mean(tf.abs(real - real_pred), axis=[1,2,3])
    gen_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1,2,3])

    loss = tf.reduce_mean(real_loss - gen_loss * gain)
    return loss


def g_loss(gen, gen_pred, original):
    # minimize L(x) - kt * L(G(v))
    # where L(v) = |v - D(v)|
    # EBGAN's discriminator as is autoencoder.

    original_loss = tf.reduce_mean(tf.square(gen - original), axis=[1,2,3])
    g_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1,2,3])

    loss = tf.reduce_mean(g_loss + original_loss)
    return loss


def balanced_d_loss(real, real_pred, gen, gen_pred, balance):
    """Calculate balanced D loss.
    """
    real_loss = tf.reduce_mean(tf.abs(real - real_pred), axis=[1,2,3])
    gen_loss = tf.reduce_mean(tf.abs(gen - gen_pred), axis=[1,2,3])

    loss = tf.reduce_mean(balance * real_loss - gen_loss)
    return loss


def global_measure(real, real_pred, balanced_loss):
    # global convergence is calculated by |D(v) - D(G(v))|
    d_loss = tf.reduce_mean(tf.abs(real - real_pred), axis=[1,2,3])
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
        train_step = optimizer.minimize(
            loss, global_step=self.global_step, var_list=var_list)

        return train_step
