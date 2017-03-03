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
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.utils import np_utils
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import InputLayer, MaxPooling2D, Convolution2D
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import argparse
import os

# === prepare keras network === #

#config setting
batch_size = 128
nb_classes = 10
nb_epoch = 6


f_model = './models_keras'
model_filename = 'model.yaml'
weights_filename = 'model_weights.hdf5'


imageDim = 784
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
    X_train = X_train.reshape(X_train.shape[0], imageDim)
    X_test = X_test.reshape(X_test.shape[0], imageDim)
    input_shape = (imageDim,)
    #print(X_train.shape)
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
        model.load_weights(os.path.join(f_model, weights_filename))
        trained = True
    else:
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape, name='input'))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax', name='softmax'))

    optimizer = SGD(lr=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("Keras network already prepare done...")
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    if not trained:
        print('Training new network...')
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
        
        # Save checkpoint, graph.pb and tensorboard
        saver.save(sess, f_model + "/model.ckpt")
        tf.train.write_graph(sess.graph.as_graph_def(), f_model, "graph.pb")
        tf.summary.FileWriter("board", sess.graph)
        
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, model_filename), 'w').write(yaml_string)
        model.save_weights(os.path.join(f_model, weights_filename))
        
    [print(n.name) for n in sess.graph.as_graph_def().node]

    start = time.perf_counter()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('score:', score[0])
    print('accuracy:', score[1])
    print('elapsed time for test {} [msec]'.format((time.perf_counter()-start) * 1000))

    start = time.perf_counter()
    n_loop = 5
    for n in range(n_loop):
        [model.predict(np.array([x])) for x in X_test]
    print('elapsed time for {} prediction {} [msec]'.format(len(X_test), (time.perf_counter()-start) * 1000 / n_loop))

    pred = K.function([model.input], [model.output])
    start = time.perf_counter()
    for n in range(n_loop):
        [pred([np.array([x])]) for x in X_test]
    print('elapsed time for {} prediction {} [msec]'.format(len(X_test), (time.perf_counter()-start) * 1000 / n_loop))
    
    KTF.set_session(old_session)

    print('-' * 30)
    print('total elapsed time {} [msec]'.format((time.perf_counter()-total_start) * 1000))

