import numpy as np
import pandas as pd
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=111
)

print(np.min(x_train), np.max(x_train)) # 0.0 711.0
print(np.min(x_test), np.max(x_test))   # 0.0 666.0

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (404, 13) (404,)
print(x_test.shape, y_test.shape)   # (102, 13) (102,)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

print(x_train.shape, y_train.shape) # (404, 13, 1, 1) (404,)
print(x_test.shape, y_test.shape)   # (102, 13, 1, 1) (102,)
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # 0.0 1.0

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, 2, input_shape=(13,1,1), padding='same'))
model.add(Conv2D(32, 2, padding='same'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 32,
          verbose=1, validation_split=0.2, callbacks=es)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)
print('runtime : ', round(end_time - start_time, 2))

# loss :  35.94442367553711
# 0.6036698195413819
# runtime :  1.14