// **************************************************************************
// MIT License
//
// Copyright (c) [2016-2018] [Jacky-Tung]
// (2017 convert to keras conv example by YukiB)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ***************************************************************************

// ---------------------------------------------------------------------------------
// The mnist.cc is a example to show how to use tensorflow c++ api to load the graph
// and do the prediction. Based on "jim-fleming/loading a tensorflow graph with the
// c plus api", I add much more complicate example. After finish reading the code, I
// hope you can understand
// 1. how to load graph
// 2. how to declare a tensor and put data to tensor correctly
// 3. how to read output tensor and do the prediction
//
// Reference:
// 1. load mnist data is reference by https://github.com/krck/MNIST_Loader
// 2. how to use tensorflow c++ api to load the graph is reference by 
// https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.chz3r27xt
// ---------------------------------------------------------------------------------
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "MNIST.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;
using namespace tensorflow;

int main(int argc, char* argv[]) {
  
  std::chrono::system_clock::time_point  total_start;
  total_start = std::chrono::system_clock::now();

  // Initialize a tensorflow session
  cout << "start initalize session" << "\n";
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  const char* file_name;
  const char* default_file_name = "./frozen_graph_keras.pb";
  
  const char* input_name;
  const char* default_input_name = "input";
  
  const char* output_name;
  const char* default_output_name = "Softmax";
  
  if (argc == 4) {
    file_name = argv[1];
    input_name = argv[2];
    output_name = argv[3];
  } else {
    file_name = default_file_name;
    input_name = default_input_name;
    output_name = default_output_name;
  }

  cout << "load " << file_name << endl;
  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), file_name, &graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }
  cout << "loaded graph" << "\n";
  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }
  
  cout << "preparing input data..." << endl;
  // config setting
  int imageDim = 784;
  int nTests = 10000;
  
  // Setup inputs and outputs:
  Tensor x(DT_FLOAT, TensorShape({nTests, imageDim}));

  MNIST mnist = MNIST("./MNIST_data/");
  auto dst = x.flat<float>().data();
  for (int i = 0; i < nTests; i++) {
    auto img = mnist.testData.at(i).pixelData;
    std::copy_n(img.begin(), imageDim, dst);
    dst += imageDim;
  }

  cout << "data is ready" << endl;
  vector<pair<string, Tensor>> inputs = {
    {input_name, x}
  };

  cout << "run test" << endl;
  
  std::chrono::system_clock::time_point  start, end;
  start = std::chrono::system_clock::now();
  
  // The session will initialize the outputs
  vector<Tensor> outputs;
  // Run the session, evaluating our "softmax" operation from the graph
  status = session->Run(inputs, {output_name}, {}, &outputs);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }else{
  	cout << "Success load graph !! " << "\n";
  }

  // start compute the accuracy,
  // arg_max is to record which index is the largest value after 
  // computing softmax, and if arg_max is equal to testData.label,
  // means predict correct.
  
  int nHits = 0;
  for (vector<Tensor>::iterator it = outputs.begin() ; it != outputs.end(); ++it) {
  	auto items = it->shaped<float, 2>({nTests, 10}); // 10 represent number of class
	for(int i = 0 ; i < nTests ; i++){
	     int arg_max = 0;
      	     float val_max = items(i, 0);
      	     for (int j = 0; j < 10; j++) {
        	if (items(i, j) > val_max) {
          	    arg_max = j;
          	    val_max = items(i, j);
                }
	     }
	     if (arg_max == mnist.testData.at(i).label) {
        	 nHits++;
      	     } 
	}
  }
  float accuracy = (float)nHits/nTests;
  cout << "accuracy is : " << accuracy << ", and Done!!" << "\n";
  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  cout << "elapsed time " << elapsed / 1000.0 << "[msec]" << endl;

  // Predition test
  start = std::chrono::system_clock::now();
  for (int i = 0; i < nTests; i++) {
    Tensor minix(DT_FLOAT, TensorShape({1, imageDim}));
    auto dst = minix.flat<float>().data();
    auto img = mnist.testData.at(i).pixelData;
    std::copy_n(img.begin(), imageDim, dst);
    vector<pair<string, Tensor>> miniInputs = {
      {input_name, minix}
    };
    vector<Tensor> miniOutputs;
    status = session->Run(miniInputs, {output_name}, {}, &miniOutputs);
    if (!status.ok()) {
      cout << status.ToString() << "\n";
      return 1;
    }
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  cout << "elapsed time for " << nTests <<  " prediction " << elapsed / 1000.0 << "[msec]" << endl;

  
  double total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-total_start).count();
  cout << "total elapsed time " << total_elapsed / 1000.0 << "[msec]" << endl;
  return 0;
}
