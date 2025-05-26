# keras18-1 copy

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston

plt.rcParams['font.family'] = 'Malgun Gothic'

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
model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',         # 모니터 할 값
    mode='min',                 # 최대값 max, 알아서 찾기 auto
    patience=40,                # 40번 안에 최소값이 안나오면 멈춤
    restore_best_weights=True,  # 최소값을 저장할거면 True, default는 False
    verbose=1
    )
hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 32,
          verbose=1, validation_split=0.2, callbacks=[es])

#region hist에 epoch만큼의 loss와 val_loss가 저장된다
print('============================hist============================')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000206D6717B50>
print('========================hist.history========================')
print(hist.history)
print('============================loss============================')
print(hist.history['loss'])
print('==========================var_loss==========================')
print(hist.history['val_loss'])
#endregion

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)
print(r2)

# pationce 40, epochs 143
# 10.535536766052246
# 0.8727698822270811

# 그래프
plt.figure(figsize=(9, 6))      # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간순으로 그림을 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('Boston Loss 보스턴 로스')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')   # 우측 상단에 label표시
plt.grid()                      # 격자 표시
plt.show()