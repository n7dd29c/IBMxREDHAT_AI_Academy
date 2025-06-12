from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
print(x_train.shape, x_test.shape)  # (50000, 3072) (10000, 3072)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Dense(4096, input_dim=3072, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3072, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
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
model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])
print(results[1])
print(end_time-start_time)

# 1.3969573974609375
# 0.5185999870300293
# 260.4936192035675