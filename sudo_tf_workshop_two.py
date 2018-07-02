"""
Working with initializations, datasets and more

Placeholders work for simple experiments, but Datasets are the preferred method of streaming data into a model
"""

import tensorflow as tf 
import numpy as np 

my_data = [[1, 2],
			[3, 4],
			[5, 6],
			[7, 8],
			[9, 10]]

# A sample dataset or 2x5 shape. Note that it is not an nupmy array or tf.Constant

slices = tf.data.Dataset.from_tensor_slices(my_data)
# slices is tf.data.iterator type object
# we use the make_one_shot_iterator and get_next methods of "slices" object
print(slices)
next_item = slices.make_one_shot_iterator().get_next()
print(next_item)

# If the end of the dataset, slices is reached, it thrown an out_of_range error
sess = tf.Session()

while True:
	try:
		print(sess.run(next_item))
	except tf.errors.OutOfRangeError:
		break



item_iterator = slices.make_one_shot_iterator()
print(item_iterator)

get_next = item_iterator.get_next()
print(get_next)


"""
If the Dataset depends on stateful operations you may need to initialize the iterator before using it, as shown below:
"""
print("\n----------------------------------------------------------\n")

data = tf.random_normal(shape = [10, 3])
print(data)

slices = tf.data.Dataset.from_tensor_slices(data)
print(slices)

iterator = slices.make_initializable_iterator()
print(iterator)

get_next = iterator.get_next()

print("What is this initializer?")
print(sess.run(iterator.initializer)) # What is this initializer? Because, it is a stateful operator
# Stateful operators need to be initialized before use: https://www.tensorflow.org/api_guides/python/constant_op#Random_Tensors

while True:
	try:
		print(sess.run(get_next))
	except tf.errors.OutOfRangeError:
		break

sess.close() # Always manually close a session


with tf.Session() as sess:
	print("Alternatively we can always do this")
	print(sess.run(data))
	print(my_data)




"""
Layers

A trainable model must modify the values in the graph to get new outputs with the same input. Layers are the preferred way to 
add trainable parameters to a graph.

Layers package together both the variables and the operations that act on them. For example a densely-connected layer performs a 
weighted sum across all inputs for each output and applies an optional activation function. The connection weights and biases are 
managed by the layer object.
"""
print("\n\n----------------------------------------------------------")

x = tf.placeholder(dtype = tf.float32, shape = [None, 3], name = "inputLayer")
print(x)

linear_model = tf.layers.Dense(units = 1) # units = no output classes, here units = 1
print(linear_model)

y = linear_model(x) # Weighted sum of all inputs and creates a single output
print(y)

init = tf.global_variables_initializer()
print("init type", init)

sess = tf.Session()
print(sess)

sess.run(init)
print(sess.run(y, feed_dict = {x: sess.run(data)})) # Nesting passing a sess.run() value to another sess.run(), both are different sessions


print("\nMerging the layers object with the output tensor creates a shosrtcut")
y = tf.layers.dense(x, units = 1)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, feed_dict = {x: sess.run(data)}))

sess.close()


"""
Type of tf.Tensors

	---> tf.Cnstants
	---> tf.random
	----> tf.sequence
	"""


# For tensorboard
writer = tf.summary.FileWriter(".")
writer.add_graph(tf.get_default_graph())