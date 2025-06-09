import sklearn as sk
import tensorflow as tf
import numpy as np
import time
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 인증서 오류 처리

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,               # 각각 train과 test에 들어갈 값
    test_size=0.1,      # 전체 데이터 중 테스트데이터에 쓸 비율, 학습데이터는 자동으로 나머지로 정해짐
    random_state=748    # 랜덤시드값
)

scaler = RobustScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(16, input_dim = 8, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

input = Input(shape=(8,))
dense1 = Dense(16, activation='relu')(input)
dense2 = Dense(32, activation='relu')(dense1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(32, activation='relu')(drop5)
drop6 = Dropout(0.2)(dense6)
dense7 = Dense(32, activation='relu')(drop6)
dense8 = Dense(16, activation='relu')(dense7)
output = Dense(1)(dense8)
model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
          verbose=1, validation_split=0.1)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)
print('runtime : ', round(end_time - start_time, 2))

# 0.7980838853379748

# 0.7050866891421564