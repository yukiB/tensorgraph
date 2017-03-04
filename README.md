# tf-keras-speed-test
This is an **keras and tensorflow example** to show the difference of　a model's execution speed between keras and tensorflow, executed from python and c++. 

This code forked from JackyTung/tensorgraph.
More description can check on JackyTung's [blog post](http://jackytung8085.blogspot.tw/2016/06/loading-tensorflow-graph-with-c-api-by.html)</br>


# Requirement
- tensorflow installation, https://www.tensorflow.org/ <br> go to "GET STARTED" --> "installing from source"
- bazel installation, http://www.bazel.io/docs/install.html <br>

# gengraph
How to generate checkpoint, graph.pb, tensorboard. <br>
The directory struct is
```
mnist_tf.py
mnist_keras_tf.py
mnist_keras.py
mnist_tf_conv.py
mnist_keras_tf_conv.py
mnist_keras_conv.py
board/
```
After run
```
$ python mnist_tf.py
```
The directory struct will be expected to
```
mnist_tf.py
...
board/
    event.out.tfevents
models_tf/
    graph.pb
    model.ckpt
Mnist_data/
    ...
```
# generate frozen graph
From Tensorflow official guide says that:

What this does is load the GraphDef, pull in the values for all the variables from the latest checkpoint file, and then replace each Variable op with a Const that has the numerical data for the weights stored in its attributes It then strips away all the extraneous nodes that aren't used for forward inference, and saves out the resulting GraphDef into an output file

Hence, we do the following steps to generate frozen graph
```
bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=graph.pb \
--input_checkpoint=model.ckpt \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax
```

# loadgraph
How to load graph with tensorflow c++ api and do the prediction. <br>
Put the directory to tensorflow source code.
Here is the final directory structure:
```
tensorflow/tensorflow/loadgraph
tensorflow/tensorflow/loadgraph/mnist.cc
tensorflow/tensorflow/loadgraph/MNIST.h
tensorflow/tensorflow/loadgraph/BUILD
```
Compile and Run
```
From inside the project folder call $bazel build :mnistpredict
From the repository root, go into bazel-bin/tensorflow/loadgraph.
Copy the frozen_graph.pb and Mnist_data to bazel-bin/tensorflow/loadgraph
Then run ./mnistpredict and check the output
```
# Reference
[MNIST_Loader](https://github.com/krck/MNIST_Loader) <br>
[Load graph with tensorflow c++ api](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.chz3r27xt)


