import tensorflow as tf


a=tf.constant([1,2,3,4,5,6,7,8,9])
condition=tf.less(a,5)
result=tf.where(condition, a*3, a*2)

with tf.Session() as sess:
    print (sess.run(result))