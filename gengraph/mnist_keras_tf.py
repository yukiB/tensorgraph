# **************************************************************************
# MIT License
#
# Copyright (c) [2017-2018] [YukiB]
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
# The mnist_keras.py is based on mnist beginner example from keras tutorial.
# In this example, we show
# 1. use keras to do the mnist data prediction
# 2. save three models (checkpoint, graph.pb, tensorboard) when execute tensorflow
# ----------------------------------------------------------------------------
import tensorflow as tf
import time
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation
from keras.layers import InputLayer
from tensorflow.examples.tutorials.mnist import input_data
import keras.backend.tensorflow_backend as KTF
import argparse
import os
import random

# === prepare keras network === #

#config setting
batch_size = 128
nb_classes = 10
nb_epoch = 6


f_model = './models_keras_tf'
model_filename = 'model.yaml'
weights_filename = 'model_weights.hdf5'


imageDim = 784
input_shape = (imageDim,)


if __name__ == "__main__":
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force-train", dest="force_train", action="store_true", default=False, help="train new model forcibly although there is a trained model (default: False)")
    args = parser.parse_args()

    if not os.path.isdir(f_model):
        os.makedirs(f_model)    
    
    old_session = KTF.get_session()
    sess = tf.Session()
    KTF.set_session(sess)
    
    trained = False
    checkpoint = tf.train.get_checkpoint_state(f_model)


    if (not args.force_train) and checkpoint and checkpoint.model_checkpoint_path:
        yaml_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_yaml(yaml_string)
        trained = True
    else:
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape, name='input'))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax', name='softmax'))

    x = tf.placeholder(tf.float32, [None, imageDim], name="input")
    y = model(x)
    y_ = tf.placeholder(tf.float32, [None, nb_classes])

    # cross-entropy
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    print("tensorflow network already prepare done...")
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    tfmnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if trained:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            if i % 100 == 0:
                print("iteration num :", i)
            batch_xs, batch_ys = tfmnist.train.next_batch(100)
            sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})


        # Save checkpoint, graph.pb and tensorboard
        saver.save(sess, f_model + "/model.ckpt")
        tf.train.write_graph(sess.graph.as_graph_def(), f_model, "graph.pb")
        tf.summary.FileWriter("board", sess.graph)
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, model_filename), 'w').write(yaml_string)

        
    #Testing
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    start = time.perf_counter()
    print(
        "accuracy : ",    
        sess.run(
            accuracy,
            feed_dict={x: tfmnist.test.images, y_:tfmnist.test.labels}
        )
    )
    print('elapsed time {} [msec]'.format((time.perf_counter()-start) * 1000))

    start = time.perf_counter()
    n_loop = 5
    for n in range(n_loop):
        [sess.run(y, feed_dict={x: np.array([test_x])}) for test_x in tfmnist.test.images]
    print('elapsed time for {} prediction {} [msec]'.format(len(tfmnist.test.images), (time.perf_counter()-start) * 1000 / n_loop))
    print('-' * 30)
    KTF.set_session(old_session)

    print('total elapsed time {} [msec]'.format((time.perf_counter()-total_start) * 1000))

