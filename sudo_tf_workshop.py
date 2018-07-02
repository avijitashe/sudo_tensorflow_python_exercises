"""
TensorFlow Workshop

https://www.tensorflow.org/programmers_guide/low_level_intro

To get the most out of this guide, you should know the following:

    How to program in Python.
    At least a little bit about arrays.
    Ideally, something about machine learning.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

# Creating simple tensors
x = np.array(3)
print ("x.shape: scalar: rank 0", x.shape)


x = np.array([3])
print ("x.shape: vector: rank 1", x.shape)

x = np.array([3., 4., 5.])
print ("x.shape: vector: rank 1", x.shape)

x = np.array([[3]])
print ("x.shape: vector: rank 2", x.shape)
print("Note: Always use this notation for any colum or row vector!")


x = np.array([[3., 4., 5.]])
print ("x.shape: vector: rank 2", x.shape)
print("Note: Always use this notation for any colum or row vector!")


x = np.array([[3., 4., 5.], [3., 4., 5.]])
print ("x.shape: vector: rank 2", x.shape)


print("\n\n")

a = tf.constant(3.0, dtype = tf.float32, name = "simple3")
print(a)

b = tf.constant(3.0) 
print(b)

# Each element in the computation graph is known by a name. Useful for TensorBoard
# https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard

print(a + b)

total = a + b
print(total)

total = tf.add(a, b, name = "simple_total")
print(total)


# Saving the computation graph for visualization with TensorBoard

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph()) # Every time to execute this .py file, a new event is written and time-stapped

# This saves an event in the present directory
# Use: "tensorboard --logdir ." to launch and view the computation graph, yes a dot after logdir
# dot means, present/current directory

sess = tf.Session()
output = sess.run(total)
print(output) 
sess.close()



with tf.Session() as sess:
	print("Total: " + str(sess.run(total))) # str() type casts int/float to char type
	print("We don't need sess.close. It does it implicitly.")


with tf.Session() as sess:
	print("\n\nA single tf.Session can accept multiple tf.Tensors")
	print(sess.run({"a,b:": (a,b), "total": total }))
	print("The tf.Tensors a, b and total were overwritten with the feed_dict names passed here")
	print(sess.run({"oh my god": (a, b), "so was me": total}))
	print("Why? As the were not placeholders, just random tf.Tensors. For placeholders, names matter")



v = tf.random_uniform(shape=(3,1)) # Rank 2
out1 = v + 1
out2 = v + 2
with tf.Session() as sess:
	print("\n\nHoweve, a tf.Tensor can have only single value inside a tf.Session.run")
	print("v 1st time", sess.run(v))
	print("v 2nd time", sess.run(v))
	print("v + 1, v + 2", sess.run((out1, out2)))




v = tf.random_uniform(shape=(3,)) # Rank 1
out1 = v + 1
out2 = v + 2
with tf.Session() as sess:
	print("\n\nThis time with a rank 1 vector")
	print("v 1st time", sess.run(v))
	print("v 2nd time", sess.run(v))
	print("v + 1, v + 2", sess.run((out1, out2)))


# Plceholders
print("\n\nWith placeholders")
x = tf.placeholder(tf.float32, name = "place1")
y = tf.placeholder(tf.float32, name = "place2")
total = x + y

with tf.Session() as sess:
	print(sess.run(total, feed_dict={x:5.4, y:2.7}))
	print(sess.run(total, feed_dict={x:9.45, y:25.6}))
	print(sess.run(total, feed_dict={x:[1, 2.3], y:[23, 2.3]})) # Implicit call to tf.add()


