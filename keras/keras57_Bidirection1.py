import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)          # (7, 3, 1)

#2. 모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

# -------------------------------
# LSTM                      
# Total params: 565         
# Trainable params: 565     
# Non-trainable params: 0   

# LSTM(Bidirectional)
# Total params: 1,115
# Trainable params: 1,115
# Non-trainable params: 0
# -------------------------------
# SimpleRNN
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0

# SimpleRNN(Bidirectional)
# Total params: 395
# Trainable params: 395
# Non-trainable params: 0
# -------------------------------
# GRU
# Total params: 475
# Trainable params: 475
# Non-trainable params: 0

# GRU(Bidirectional)
# Total params: 935
# Trainable params: 935
# Non-trainable params: 0
# -------------------------------
