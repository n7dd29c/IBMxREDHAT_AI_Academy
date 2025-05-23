import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston
print(sk.__version__)

#1. 데이터
dataset = load_boston()
x = dataset.data    
y = dataset.target  

# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=222
)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim = 13))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 700, batch_size = 32, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)
print(r2)
# 0.7432790936563991