"""
Feature columns

The easiest way to experiment with feature columns is using the tf.feature_column.input_layer function. 
This function only accepts dense columns as inputs, so to view the result of a categorical column you must 
wrap it in an tf.feature_column.indicator_column

"""

import tensorflow as tf 
import numpy as np 

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

# We have two features here as matrices: 1st numeric, 2nd categorical non-numeric

print(features)
print("features data type: ", type(features)) # Visually: a dict or dictionary because of parenthesis {}

# How to represent non-numeric categorical columns in tensorflow?
# We use one-hot encoding and create dense, numeric feature columns from them

department_column = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sports', 'gardening'])
# It converts it into a categorical column from just a feature with non-numeric data
print("\n")
print("department_column type: ", department_column)


print("\n")
now_department_column = tf.feature_column.indicator_column(department_column)
print("now_department_column type: ", now_department_column)

# Now features are converted to columns that can be directly fed into the tensorflow for computation

columns = [
			tf.feature_column.numeric_column('sales'),
			now_department_column ]
print("\n")
print("columns type: ", columns)

# Converion into one-hot encoding or numerica data is not complete
inputs = tf.feature_column.input_layer(features, columns) # Shoould be tf.Tensor type, a one hot encoding, with (4, 3)

# Why 3? 1 for sales, 2 for category sports, 3 for cateogry gardening
# Why 4? There are 4 samples in the above dataset

print("\n")
print("inputs type: ", inputs)


# Intializations
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

print("\n")

with tf.Session() as sess:
	sess.run((var_init, table_init))
	print(sess.run(inputs))


# Now, we can use this input as a dataset and feed it to neural network or anything else
# It is a 3 feature, 4 obs dataset
# It is a Rank 2 tensor