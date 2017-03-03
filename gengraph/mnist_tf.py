# **************************************************************************
# MIT License
#
# Copyright (c) [2016-2018] [Jacky-Tung]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ***************************************************************************

# ----------------------------------------------------------------------------
# The mnist.py is based on mnist beginner example from tensorflow tutorial.
# In this example, we show
# 1. use tensorflow to do the mnist data prediction
# 2. save three models (checkpoint, graph.pb, tensorboard) when execute tensorflow
# ----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os

# === prepare tensorflow network === #

f_model = "./models_tf"

#config setting
imageDim = 784
outputDim = 10

if __name__ == "__main__":
    # args
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force-train", dest="force_train", action="store_true", default=False, help="train new model forcibly although there is a trained model (default: False)")
    args = parser.parse_args()
    
    if not os.path.isdir(f_model):
        os.makedirs(f_model)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # None means that a dimension can be any of any length
    x = tf.placeholder(tf.float32, [None, imageDim], name="input")
    # 784-dimensional image vectors by it to produce 10-dimensional vectors
    W = tf.Variable(tf.zeros([imageDim, outputDim]), dtype=tf.float32, name="Weight")
    # a shape of [10]
    b = tf.Variable(tf.zeros([outputDim]), dtype=tf.float32, name="bias")
    # softmax
    y = tf.nn.softmax(tf.matmul(x, W)+b, name="softmax")
    # print(x, W, b, y)

    # input correct answers
    y_ = tf.placeholder(tf.float32, [None, outputDim])

    # cross-entropy
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    trained = False

    print("tensorflow network already prepare done...")

    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(f_model)
        if (not args.force_train) and checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
            trained = True
        else:
            print('Training new network...')
            sess.run(tf.global_variables_initializer())

        #Training
        if not trained:

            for i in range(1000):
                if i % 100 == 0:
                    print("iteration num :", i)
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})

            # Save checkpoint, graph.pb and tensorboard
            saver.save(sess, f_model + "/model.ckpt")
            tf.train.write_graph(sess.graph.as_graph_def(), f_model, "graph.pb")
            tf.summary.FileWriter("board", sess.graph)

        #Testing
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        start = time.perf_counter()
        print(
            "accuracy : ",    
            sess.run(
                accuracy,
                feed_dict={x: mnist.test.images, y_:mnist.test.labels}
            )
        )
        print('elapsed time {} [msec]'.format((time.perf_counter()-start) * 1000))

        start = time.perf_counter()
        n_loop = 5
        for n in range(n_loop):
            [sess.run(y, feed_dict={x: np.array([test_x])}) for test_x in mnist.test.images]
        print('elapsed time for {} prediction {} [msec]'.format(len(mnist.test.images), (time.perf_counter()-start) * 1000 / n_loop))
        print('-' * 30)
        print('total elapsed time {} [msec]'.format((time.perf_counter()-total_start) * 1000))


