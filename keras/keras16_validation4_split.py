# keras16-3 copy

import tensorflow as tf
import numpy as np
import sklearn as sk

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array(range(1, 17))
y = np.array(range(1, 17))

# # 리스트 슬라이싱으로 10:3:3으로 나눈다

# x_train = x[:10]
# y_train = y[:10]
# print(x_train, y_train)

# x_val = x[9:13]
# y_val = y[9:13]
# print(x_val, y_val)

# x_test = x[13:]
# y_test = y[13:]
# print(x_test, y_test)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.85,
    random_state=111
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=90, batch_size=1,
          verbose=1, validation_split=0.2
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
results = model.predict([17])
print('[17]의 예측값 : ', results)