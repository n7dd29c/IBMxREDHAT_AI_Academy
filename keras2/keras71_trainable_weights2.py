import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])

x = np.array([1])
y = np.array([1])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

########################### 동결 ###########################
model.trainable = False     # 동결
# model.trainable = True      # 동결하지 않음, defalut

print('===================================================================================')
print(model.weights)
print('=========================== =======================================================')

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, verbose=0)

y_pred = model.predict(x)
print(y_pred)

print(model.weights)