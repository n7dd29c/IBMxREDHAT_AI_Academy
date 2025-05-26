import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

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
hist = model.fit(x_train, y_train, epochs = 700, batch_size = 32,
          verbose=3, validation_split=0.2)
# hist에 epoch만큼의 loss와 val_loss가 저장된다
print('============================hist============================')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000206D6717B50>
print('========================hist.history========================')
print(hist.history)
print('============================loss============================')
print(hist.history['loss'])
print('==========================var_loss==========================')
print(hist.history['val_loss'])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)
print(r2)
# 0.8498238732697008

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