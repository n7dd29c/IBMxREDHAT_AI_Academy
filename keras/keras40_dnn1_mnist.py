# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (60000,)
print(y_train.shape, y_test.shape)  # (10000, 28, 28) (10000,)

''' 스케일링 '''
x_train = x_train/255.
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(x_train.shape, x_test.shape)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(-1, 1)
# print(y_train.shape, y_test.shape)  # (60000, 1) (10000, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# to_categorical 사용
# y_train = to_categorical(y_train, 1)
# y_test = to_categorical(y_test, 1)

#2. 모델구성, 목표 acc >= 0.98, 시간 체크 (CNN과 비교)
model = Sequential()
model.add(Dense(1024, input_dim=784, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64, callbacks=es, validation_split=0.2, verbose=2)
end_time = time.time()

#4. 평가, 훈련
results = model.evaluate(x_test, y_test)
print(results[0])               # 0.08890217542648315
print(results[1])               # 0.9828000068664551
print(end_time - start_time)    # 129.15152430534363