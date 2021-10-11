#########################################################

import tensorflow as tf
import numpy as np

x_array = np.arange(18).reshape(3, 2, 3)
x2 = tf.reshape(x_array, shape = (-1, 6))

xsum = tf.reduce_sum(x2, axis=0)
xmean = tf.reduce_mean(x2, axis=0)

print(x_array.shape)
print(x2.numpy())
print(xsum.numpy())
print(xmean.numpy())

w = tf.Variable(2.0, name='weight')
b = tf.Variable(0.7, name='bias')

for x in [1.0, 0.6, -1.8]:
    z = w * x + b
    print('x=%44.1f --> z = %4.1f' % (x, z))
