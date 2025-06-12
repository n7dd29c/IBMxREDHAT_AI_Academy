from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

''' 스케일링 '''
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성
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
model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.2, callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])
print(results[1])
print(end_time-start_time)

# 0.3401563763618469
# 0.8934000134468079
# 99.18238592147827