import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x.shape)  # (1797, 64)
# print(y.shape)  # (1797,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
print(x_test.shape, y_test.shape)   # (360, 64) (360,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 8, 1)
x_test = x_test.reshape(-1, 8, 8, 1)

scaler = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

print(y_train.shape, y_test.shape)  # (1437, 10) (360, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, 2, 1, input_shape=(8, 8, 1), padding='same'))
model.add(Conv2D(32, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',         # 모니터 할 값
    mode='max',                 # 최대값 max, 알아서 찾기 auto
    patience=40,                # 40번 안에 최소값이 안나오면 멈춤
    restore_best_weights=True,  # 최소값을 저장할거면 True, default는 False
)

hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 64,
          verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
results = model.predict(x_test)

# loss :  1.2977604866027832

# dropout
# loss :  1.0834016799926758

# loss :  0.1795787662267685
# acc :  0.9722222089767456