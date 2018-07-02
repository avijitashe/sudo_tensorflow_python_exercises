"""
Tesitng basics of tf.Tensor

"""

import tensorflow as tf 

x = tf.constant([1, 2, 3])
print(tf.rank(x))
print(x.shape)

print("\nAfter reshape")
y = tf.reshape(x, [3, 1])
print(tf.rank(y))
print(y.shape)

# No values are held by x and y as tf.Tensor, so their rank is not known as of now

print("\nCreating a tf.Session to run the tf.Graph")
sess = tf.Session()
print(sess.run(x))
print("Rank: ", sess.run(tf.rank(x)))

print("\nAfter reshape")
print(sess.run(y))
print("Rank: ", sess.run(tf.rank(y)))

print("1st element: ", sess.run(x[0]))




matrix = tf.ones([10,3])
vector = matrix[3]
vectrorFromSLice = matrix[:, 2]

print("Matrix: ", sess.run(matrix))
print("Vector: ", sess.run(vector))
print("vectrorFromSLice: ", sess.run(vectrorFromSLice))

print("\nUsing tensor.eval() instead of sess.run()")
print("Matrix: ", matrix.eval(session = sess)) # Method 1


with sess.as_default():
	print("Vector: ", vector.eval())
	print("vectrorFromSLice: ", vectrorFromSLice.eval()) # Method 2


p = tf.placeholder(tf.float32)
t = p + 1.0

####################################################################################
# t.eval(session = sess)  # This will fail, since the placeholder did not get a value. 
# You should see an error here, so comment it out. Try removing comments to view the error.


t.eval(feed_dict={p:2.0}, session = sess)  # This will succeed because we're feeding a value
                           # to the placeholder.


sess.close()
