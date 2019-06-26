from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
from keras.utils import to_categorical
from keras import optimizers
import t3f

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#
# x_train = x_train / 127.5 - 1.0
# x_test = x_test / 127.5 - 1.0
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
tt_layer = t3f.nn.KerasDense(input_dims=[7, 4, 7, 4], output_dims=[5, 5, 5, 5],
tt_rank=4, activation='relu',
bias_initializer=1e-3)
model.add(tt_layer)
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
optimizer = optimizers.Adam(lr=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

