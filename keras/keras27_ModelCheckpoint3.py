import numpy as np
import pandas as pd
import sklearn as sk
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=111
)

scaler = MinMaxScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.06141956477526944 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
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

path = './_save/keras27_mcp/'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras27_mcp3.hdf5'
)

hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 64,
          verbose=1, validation_split=0.2, callbacks=[es, mcp])

model.save(path + 'keras27_3_save_model.h5')


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('R2 : ', r2)
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
rmse = RMSE(y_test, results)
print('RMSE : ', rmse)

# 11.139924049377441
# 0.9041660960806456
# RMSE :  3.337652445777287