x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential()
model.add(Dense(10, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=2000)

results = model.evaluate(x_data, y_data)
print(results)