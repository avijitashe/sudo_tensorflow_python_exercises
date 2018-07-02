"""
Training

Now that you're familiar with the basics of core TensorFlow, let's train a small regression model manually.

A dummy regeessional using tensorflow. We can also import the "inputs" from sudo_tf_workshop_three.py, instead

"""

import tensorflow as tf 
import numpy as np 

# Data
x = tf.constant([[2], [1], [3], [4]], dtype = tf.float32)
y_true = tf.constant([[4], [2], [6], [8]], dtype = tf.float32)
print("x type: ", x)
print("y_true type: ", y_true)



##########################################################
# Model: layers: Step 1
linear_model = tf.layers.Dense(units = 1)

# Output
y_pred = linear_model(x)
print("y_pred type: ", y_pred)

# Evaluate
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y_pred))

###########################################################
# No training is done yet


###########################################################
# Loss function: Step 2
loss = tf.losses.mean_squared_error(labels = y_true, predictions = y_pred)
print(sess.run(loss))
print("loss type:", loss)



###########################################################
# Training: Use an optimizer: Step 3
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.02)
train = optimizer.minimize(loss)

# Alternatively: optimizer = tf.train.GraidentDescentOptimizer(learning_rate = 0.01).minimize(loss)
print("optimizer type:", optimizer)
print("train type:", train)



print("\n\n")
###########################################################
# Iterate for some number of steps: Step 4
for i in range(200):
	_, loss_val = sess.run((train, loss))
	if i % 10 == 0:
		print("Iteration: {0}, loss = {1}".format(i, loss_val))
	
print("Traing over!!")

print("\nFinal predictions")
print(sess.run(y_pred))
print("\nActual values")
print(sess.run(y_true))


sess.close()

###########################################################
# How to decrease the error and improve the results?
#
# Intially ---------------------------
# 3 1 5 7
# 9 1 25 49
# Max 16 error in 200 iter at rate=0.02
#
# Now --------------------------------
# Check!
#
# 1) Add more observations
# 2) It works like charm!
#
# The log file for this gives the loss values for two sets of x,y pairs. It shows that more the linear relationship is,
# lesser number of observations, lesser complex and less iterations you need to learn the model. And, vice versa.
# Refer workshop_four.log