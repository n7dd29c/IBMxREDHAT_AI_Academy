import numpy as np
import pandas as pd
import sklearn as sk
import time
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 완전 비활성화

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
#  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.1, random_state=337
# )
# 이렇게 쓰면 y에 라벨값이 불균형하게 들어갈 수 있다 (line 26)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

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