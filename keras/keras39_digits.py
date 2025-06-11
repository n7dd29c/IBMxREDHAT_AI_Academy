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
# print(np.max(x), np.min(x)) # 16.0 0.0
# print(np.max(y), np.max(y)) # 9 9

# print(x.shape)  # (1797, 64)
# print(y.shape)  # (1797,)

x = x.reshape(x.shape[0], 8, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(8,8), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

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

import matplotlib.pyplot as plt
plt.imshow(x_train[3], 'gray')
plt.show()