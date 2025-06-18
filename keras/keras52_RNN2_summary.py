import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
print(x.shape)          # (7, 3, 1) // (batch_size, timesteps, feature)

# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               [[3],[4],[5]],
#               [[4],[5],[6]],
#               [[5],[6],[7]],
#               [[6],[7],[8]],
#               [[7],[8],[9]]]) 가 된다

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1), activation='relu')) # (7, 3, 1)에서 맨 앞자리만 빼주면 됨
# model.add(SimpleRNN(units=1024, input_shape=(3,1), activation='relu'))
# model.add(LSTM(units=10, input_shape=(3,1), activation='relu')) 
model.add(GRU(units=10, input_shape=(3,1), activation='relu')) 
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

# SimpleRNN
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0

# 파라미터 개수 = (units * feature) + (units * units) + (bias * units)
#                (   1   *   10  ) + (  10  *  10  ) + ( 1   *  10  ) = 120
#              = units * (feature + units + bias)
#                  10  * (   1    +  10   +  1  )


# LSTM
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0

# 파라미터 개수 = (units * feature) + (units * units) + (bias * units)
#                (   1   *   10  ) + (  10  *  10  ) + ( 1   *  10  ) = 120
#              = units * (feature + units + bias)
#                  10  * (   1    +  10   +  1  )
# LSTM은 4개의 게이트 = 120 * 4 = 480

# GRU
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 451
# Trainable params: 451
# Non-trainable params: 0