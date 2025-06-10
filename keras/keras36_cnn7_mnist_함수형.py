import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

''' 1. MinMaxScaler '''
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)
# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test))   # 255 0
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 24.0 0.0

''' 2. 0 ~ 1 사이로 변환, 정규화 (많이쓴다) '''
# x_train = x_train/255.  # 부동소수점 연산을 잘 해주기 위해 .을 붙임
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

''' 3. -1 ~ 1 사이로 변환, 정규화2 (많이쓴다) '''
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

''' 4. StandardScaler '''
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x를 reshape하기 -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1)))
# model.add(Conv2D(filters=32, kernel_size=(3,3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Flatten())      
# model.add(Dense(units=16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape=(16,)))       # input_shape는 생략가능
# model.add(Dense(units=10, activation='softmax'))

input = Input(shape=(28, 28, 1))
dense1 = Conv2D(filters=64, kernel_size=(3,3), strides=1)(input)
dense2 = Conv2D(filters=32, kernel_size=(3,3))(dense1)  
drop1 = Dropout(0.2)(dense2)
dense3 = Conv2D(32, (3,3), activation='relu')(drop1)
dense4 = Conv2D(32, (3,3), activation='relu')(dense3)
fltn = Flatten()(dense4)
dense5 = Dense(16, activation='relu')(fltn)
drop2 = Dropout(0.2)(dense5)
dense6 = Dense(10, activation='relu')(drop2)
output = Dense(10, activation='softmax')(dense6)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

date = datetime.datetime.now().strftime("%y%m%d_%H%M")
path = './_save/keras36_cnn5/'
filename = '{epoch:03d}-{val_loss:.3f}.hdf5'
filepath = ''.join([path, 'k36_', date, '_', filename])

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
y_pred = np.argmax(y_pred, axis=1)  # 한 행 마다(axis=1) 가장 높은 확률값을 뽑아내는 작업
y_test = y_test.values
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))
print(round(end_time - start_time, 0), '초')

# loss :  0.044939976185560226
# acc :  0.9860000014305115
# 0.986