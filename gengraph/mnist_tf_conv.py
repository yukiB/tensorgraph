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
import time
import numpy as np
from keras.datasets import mnist as keras_mnist
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import random
import os

# === prepare tensorflow network === #

f_model = "./models_tf_conv/"

#config setting
imageDim = 784
outputDim = 10
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32


def weight_variable(shape):    
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):      
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force-train", dest="force_train", action="store_true", default=False, help="train new model forcibly although there is a trained model (default: False)")
    args = parser.parse_args()

    if not os.path.isdir(f_model):
        os.makedirs(f_model)       

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = keras_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, outputDim)
    Y_test = np_utils.to_categorical(y_test, outputDim)    

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None,28,28,1], name="input")
    W_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1,img_rows,img_cols,1])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3,3,32,32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7*7*32,128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*32])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    	
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([128,10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2, name="softmax")


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

            for i in range(1500):
                j = random.randint(0, len(X_train - 100))
                if i % 100 == 0:
                    print("iteration num :", i)
                batch_xs = X_train[j:j+100]
                batch_ys = Y_train[j:j+100]
                #batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step,feed_dict={x: batch_xs,
                                               y_: batch_ys,
                                               keep_prob: 0.5})

            # Save checkpoint, graph.pb and tensorboard
            saver.save(sess, f_model + "model.ckpt")
            tf.train.write_graph(sess.graph.as_graph_def(), f_model, "graph.pb")
            tf.summary.FileWriter("board", sess.graph)

        #Testing
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        start = time.perf_counter()
        print(
            "Test accuracy : ",    
            sess.run(
                accuracy,
                #feed_dict={x: mnist.test.images, y_:mnist.test.labels}
                feed_dict={x: X_test, y_:Y_test, keep_prob: 1.0}
            )
        )
        print('elapsed time {} [msec]'.format((time.perf_counter()-start) * 1000))
        
        start = time.perf_counter()
        n_loop = 5
        for n in range(n_loop):
            [sess.run(y, feed_dict={x: np.array([test_x]), keep_prob: 1.0}) for test_x in  X_test]
        print('elapsed time for {} prediction {} [msec]'.format(len(X_test), (time.perf_counter()-start) * 1000 / n_loop))
        print('-' * 30)
        print('elapsed time {} [msec]'.format((time.perf_counter()-start) * 1000))

