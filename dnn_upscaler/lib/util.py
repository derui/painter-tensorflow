
import tensorflow as tf


def psnr_loss(original, gen, maximum=1.0):
    """
    calculate PSNR between original and gen
    """
    _, h, w, _ = original.shape.as_list()

    mse = tf.reduce_sum(tf.square(original - gen), axis=[1,2,3]) / (h * w)
    print(mse.shape.as_list())

    psnr = 20.0 * (tf.log(1.0 / tf.sqrt(mse)) / tf.log(10.0))

    return tf.reduce_mean(psnr)
