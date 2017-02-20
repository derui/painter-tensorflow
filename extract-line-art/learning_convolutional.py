# coding: utf-8

from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import auto_encoder as ae

import image_dataset as ds

BATCH_SIZE = 20

reader = ds.DataSetReader()
reader.prepare('./datasets')

with tf.Session() as sess:

    x = tf.placeholder("float", [None, 512, 512, 3])
    y_ = tf.placeholder("float", [None, 512, 512, 3])

    construction_op = ae.construction(x, 512, 512, 3)
    loss_op = ae.loss(y_, construction_op, x)
    training_op = ae.training(loss_op, 0.002)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./log", graph=sess.graph)
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state('./log')
    if ckpt:
        last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
        print("load {}".format(last_model))
        saver.restore(sess, last_model)  # 変数データの読み込み
    else:
        sess.run(tf.global_variables_initializer())

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(20000):
        batch = reader.read_batch(BATCH_SIZE)
        shape = [BATCH_SIZE, 512, 512, 3]

        feed = {
            x: np.reshape(batch[1], shape),
            y_: np.reshape(batch[0], shape)
        }
        sess.run(
            training_op,
            feed_dict=feed,
            run_metadata=run_metadata,
            options=run_options)

        if (i + 1) % 10 == 0:
            print('step {}, time:{}'.format(i + 1, datetime.utcnow().isoformat()))

        if (i + 1) % 100 == 0:
            summary_str = sess.run(summary, feed_dict=feed)
            writer.add_summary(summary_str, i)
            saver.save(sess, './log/model.ckpt', i)

            # write train
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('./log/timeline{}.json'.format(i), 'w') as f:
                f.write(ctf)

    embeddding_var = tf.Variable('float', [None, None])
    sess.run(tf.variables_initializer([embeddding_var]))
    tf.train.Saver([embeddding_var]).save(sess, './log/model.ckpt')
