import sklearn as sk
import tensorflow as tf
import numpy as np
import time
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 인증서 오류 처리

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 완전 비활성화

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,               # 각각 train과 test에 들어갈 값
    test_size=0.1,      # 전체 데이터 중 테스트데이터에 쓸 비율, 학습데이터는 자동으로 나머지로 정해짐
    random_state=748    # 랜덤시드값
)

scaler = RobustScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
          verbose=1, validation_split=0.1)
end_time = time.time()

#4. 평가, 예측
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있어버리기~')
else:
    print('GPU 없어버리기~')
print('runtime : ', round(end_time - start_time, 2))

# GPU 있어버리기~
# runtime :  280.16

# GPU 없어버리기~
# runtime :  52.1