import numpy as np
import tensorflow as tf
import keras.backend as K
import t3f



tf.set_random_seed(0)
np.random.seed(0)
sess = tf.InteractiveSession()
K.set_session(sess)




W = t3f.random_matrix([[4, 7, 4, 7], [5, 5, 5, 5]], tt_rank=2)


print(W)


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64, [None])
initializer = t3f.glorot_initializer([[4, 7, 4, 7], [5, 5, 5, 5]], tt_rank=2)
W1 = t3f.get_variable('W1', initializer=initializer)
b1 = tf.get_variable('b1', shape=[625])
h1 = t3f.matmul(x, W1) + b1
h1 = tf.nn.relu(h1)
W2 = tf.get_variable('W2', shape=[625, 10])
b2 = tf.get_variable('b2', shape=[10])
h2 = tf.matmul(h1, W2) + b2
y_ = tf.one_hot(y, 10)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h2))


