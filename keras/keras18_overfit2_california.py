import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 인증서 오류 처리

#1. 데이터
datesets = fetch_california_housing()
x = datesets.data
y = datesets.target
# print(x.shape)  # (20640, 8)
# print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=111
)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))
rmse = RMSE(y_test, y_predict)

print('loss : ', loss)
print('RMSE : ', rmse)
print('R2 : ', r2)