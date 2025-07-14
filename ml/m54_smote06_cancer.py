# smote는 증폭을 통해 보간처리 방식

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout   
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(569, 30) (569,)
print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))
print(pd.value_counts(y))
# 1    357
# 0    212

print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=seed,
    stratify=y
)

############### SMOTE ###############
#꼭 split 하고 SMOTE 쓰기
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=seed,       
              k_neighbors=5,                  #디폴트
            #   sampling_strategy='auto',     #디폴트
            #   sampling_strategy=0.75,       #최대값의 75% 지정
              sampling_strategy={0:1000, 1:1000}  #직접 지정
            #   n_jobs=-1, #0.13에서는 삭제됨. 그냥 포함됨
              )
x_train , y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([1000, 1000]))

#2. 모델
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer = 'adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape) # (143, 1)
y_pred = (y_pred > 0.5).astype(int).reshape(-1) #이진분류 일 때
print(y_pred.shape) # (143,)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='binary') # f1은 default로 binary만 받기 때문에 다중일 경우 macro 써줘야됨
print('accuracy_score: ', acc)
print('f1_score: ', f1)

############### 결과 ###############
#1. 변환하지 않은 원 데이터 훈련
# accuracy_score:  0.6293706293706294
# f1_score:  0.7725321888412017

#2. 2번에서 SMOTE 적용
# accuracy_score:  0.6293706293706294
# f1_score:  0.7725321888412017

#3. 재현's SMOTE
# accuracy_score:  0.9230769230769231
# f1_score:  0.9411764705882353
