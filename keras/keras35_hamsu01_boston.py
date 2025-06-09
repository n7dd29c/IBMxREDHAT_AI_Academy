import numpy as np
import pandas as pd
import time
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(datasets.DESCR)
# print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=111
)

scaler = MinMaxScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train)) # 0.0 1.0
# print(np.min(x_test), np.max(x_test))   # -0.06141956477526944 1.0

#2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=13, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='relu'))

input1 = Input(shape=(13,))
dense1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(128, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(64, activation='relu')(drop4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
          verbose=1, validation_split=0.1)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results)
print('runtime : ', round(end_time - start_time, 2))

# loss : 17.66684341430664
# runtime :  6.68

# loss : 14.619963645935059
# runtime :  3.71