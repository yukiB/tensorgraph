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
from keras.utils import np_utils
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import InputLayer, MaxPooling2D, Convolution2D
from tensorflow.examples.tutorials.mnist import input_data
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import argparse
import random
import os

# === prepare keras network === #

#config setting
batch_size = 128
nb_classes = 10
nb_epoch = 6


f_model = './models_keras_tf_conv'
model_filename = 'model.yaml'
weights_filename = 'model_weights.hdf5'


# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)



if __name__ == "__main__":
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force-train", dest="force_train", action="store_true", default=False, help="train new model forcibly although there is a trained model (default: False)")
    args = parser.parse_args()

    if not os.path.isdir(f_model):
        os.makedirs(f_model)    

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    old_session = KTF.get_session()
    sess = tf.Session('')
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
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax', name='softmax'))
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1], name='input')
    y = model(x)
    y_ = tf.placeholder(tf.float32, [None, nb_classes])

    # cross-entropy
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    print('tensorflow network already prepare done...')
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    K._LEARNING_PHASE = tf.constant(1)

    if trained:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print('Training new network...')
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            j = random.randint(0, len(X_train - 100))
            if i % 100 == 0:
                print('iteration num :', i)
            batch_xs = X_train[j:j+100]
            batch_ys = Y_train[j:j+100]
            sess.run(train_step,feed_dict={x: batch_xs,
                                           y_: batch_ys,
                                           K.learning_phase(): 1})
        
        # Save checkpoint, graph.pb and tensorboard
        saver.save(sess, f_model + '/model.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), f_model, 'graph.pb')
        tf.summary.FileWriter('board', sess.graph)
        
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, model_filename), 'w').write(yaml_string)
        
    # [print(n.name) for n in sess.graph.as_graph_def().node]
    
    K._LEARNING_PHASE = tf.constant(0)
    start = time.perf_counter()
    #Testing
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    start = time.perf_counter()
    print(
        'Test accuracy : ',    
        sess.run(
            accuracy,
            feed_dict={x: X_test, y_:Y_test, K.learning_phase(): 0}
        )
    )
    print('elapsed time {} [msec]'.format((time.perf_counter()-start) * 1000))
    
    start = time.perf_counter()
    n_loop = 5
    for n in range(n_loop):
        [sess.run(y, feed_dict={x: np.array([test_x]), K.learning_phase(): 0}) for test_x in X_test]
    print('elapsed time for {} prediction {} [msec]'.format(len(X_test), (time.perf_counter()-start) * 1000 / n_loop))
    
    KTF.set_session(old_session)

    print('total elapsed time {} [msec]'.format((time.perf_counter()-total_start) * 1000))
