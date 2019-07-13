import numpy as np
import tensorflow as tf


img = [0,1,2,3,4,5,6,7,8,9]
lbl = [0,1,2,3,4,5,6,7,8,9]
images = tf.convert_to_tensor(img)
labels = tf.convert_to_tensor(lbl)
input_queue = tf.train.slice_input_producer([images,labels])
sliced_img = input_queue[0]
sliced_lbl = input_queue[1]

img_batch, lbl_batch = tf.train.batch([sliced_img,sliced_lbl], batch_size=3)
with tf.Session() as sess:
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(0,3): #batch size
        image_batch,label_batch = sess.run([img_batch,lbl_batch ])
        print(image_batch, label_batch)

    coord.request_stop()
    coord.join(threads)