# keras36 6 copy

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)


''' 2. 0 ~ 1 사이로 변환, 정규화 (많이쓴다) '''
x_train = x_train/255.  # 부동소수점 연산을 잘 해주기 위해 .을 붙임
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

# x를 reshape하기 -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())      
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, input_shape=(16,)))       # input_shape는 생략가능
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras37/'
filename = '.hdf5'
filepath = ''.join([path, 'k37_0', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=256, validation_split=0.2,
          callbacks=[es, mcp], verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = y_test.values
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))

import matplotlib.pyplot as plt
plt.imshow(x_train[11], 'gray')
plt.show()

# loss :  0.17612960934638977
# acc :  0.9764999747276306
# 0.9765

# 2.
# loss :  0.05428970977663994
# acc :  0.9850000143051147
# 0.985

# MaxPooling
# loss :  0.05624242499470711
# acc :  0.9919000267982483
# 0.9919