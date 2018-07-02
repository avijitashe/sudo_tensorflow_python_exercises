# sudo_tensorflow_python_exercises
It contains the .ipynb, .py and .pdf files for quick access to TensorFlow tutorials in Python as of 2018.

# TensorFlow Core Python Exercises

We get started with the low-level programming practise from TensorFlow. Usually, it is divided into High-Level and Low-Level APIs. If you wish to learn to program your own neural nets, debug them, and create things from ground up, learn the TensorFlow Core.

This is devoid of Estimator, Import Data and Eager Execution.

## Before beginning
Source your tensorflow environment to avoid errors. To learn more about what an virtual environement is, use Google Search. Or, follow this comprehensive tensorflow installation guide.

Never forget to source your tensorflow environment before executing the python file

## Right from the beginning
To make use of TensorFlow Board, use names to variables, tensors, operations and everything you create in your computation graph.


## A Tensor!
In TensorFlow, the rank of a tensor is same as the dimension of the data (the edge) that connects to the operand (the node). The convention used here is different from the actual meaning of tensors that sprouts from Physics and Maths. If you do not know that then, it is well and good. But, if you had some familiarity with it in past, try not to confuse yourself.

The dimensionality of a matrix is same as the rank of the tensor in tensorflow.

For example: an RGB image when is fed into a CNN, it is a 4D tensor type, where the 4 dimensions being: batch_size, rows, columns, no_channels


## When launching tensorboard
Do not forget to mention the logdir as a single dot after –logdir for the present directory

For example: tensorboard –logdir .

There is a dot after –logdir above wehn typing it in the terminal.
