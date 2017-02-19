# coding: utf-8

import numpy as np
import tensorflow as tf
import auto_encoder as ae

import image_dataset as ds

BATCH_SIZE = 20

reader = ds.DataSetReader()
reader.prepare('./datasets')

with tf.Session() as sess:
    x = tf.placeholder("float", [None, 512, 512, 3])
    y_ = tf.placeholder("float", [None, 512, 512, 3])

    construction_op = ae.construction(x, 512, 512, 3)
    loss_op = ae.loss(y_, construction_op)
    training_op = ae.training(loss_op, 0.05)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./log", graph=sess.graph)
    summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = reader.read_batch(BATCH_SIZE)
        shape = [BATCH_SIZE, 512, 512, 3]

        feed = {
            x: np.reshape(batch[0], shape),
            y_: np.reshape(batch[1], shape)
        }
        sess.run(training_op, feed_dict=feed)
        if i % 200 == 0:
            summary_str = sess.run(summary, feed_dict=feed)
            writer.add_summary(summary_str, i)
            saver.save(sess, 'model.ckpt', global_step=200)
