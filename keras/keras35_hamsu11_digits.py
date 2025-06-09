import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(np.max(x), np.min(x)) # 16.0 0.0
print(np.max(y), np.max(y)) # 9 9
exit()
# print(x.shape)  # (1797, 64)
# print(y.shape)  # (1797,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

scaler = StandardScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='relu'))

input = Input(shape=(64,))
dense1 = Dense(64, activation='relu')(input)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(128, activation='relu')(drop3)
dense6 = Dense(128, activation='relu')(dense5)
dense7 = Dense(64, activation='relu')(dense6)
output = Dense(1)(dense7)

model = Model(inputs=input, outputs=output)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(
    monitor='val_loss',         # 모니터 할 값
    mode='min',                 # 최대값 max, 알아서 찾기 auto
    patience=40,                # 40번 안에 최소값이 안나오면 멈춤
    restore_best_weights=True,  # 최소값을 저장할거면 True, default는 False
)

hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 64,
          verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)

# loss :  1.2977604866027832

# dropout
# loss :  1.0834016799926758