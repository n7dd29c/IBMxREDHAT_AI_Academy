# smote는 증폭을 통해 보간처리 방식

import numpy as np
import pandas as pd
import random
import time

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout   
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=seed,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

############### SMOTE ###############
#꼭 split 하고 SMOTE 쓰기
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version: ', sk.__version__) #1.6.1
import imblearn
print('imblearn version: ', imblearn.__version__) #0.12.4

ros = RandomOverSampler(random_state=seed,
                        sampling_strategy={1:300000, 2:300000, 3:300000, 4:300000, 5:300000, 6:300000, 7:300000}  #직접 지정
                        )

x_train , y_train = ros.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([267, 267], dtype=int64))

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

#2. 모델
model = Sequential()
model.add(Dense(256, input_dim=54, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam', metrics=['acc']) 
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=3, validation_split=0.2, callbacks=[es])
end = time.time()


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape) # one-hot 형태로 나옴
y_pred = np.argmax(y_pred, axis=1) #다중분류일 때 
y_test = np.argmax(y_test, axis=1)
# y_pred = (y_pred > 0.5).astype(int).reshape(-1) #이진분류일 때
print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro') # f1은 default로 binary만 받기 때문에 다중일 경우 macro 써줘야됨
print('accuracy_score: ', acc)
print('f1_score: ', f1)
print("걸린시간:", round(end-start,2),'초')

############### 결과 ###############
#1. 변환하지 않은 원 데이터 훈련
# accuracy_score:  0.8307160609419427
# f1_score:  0.6962048560308798
# 걸린시간: 166.14 초

#2. 2번에서 SMOTE 적용
# accuracy_score:  0.8103034016509125
# f1_score:  0.7479971193506482
# 걸린시간: 555.81 초

#3. 재현's SMOTE
# accuracy_score:  0.8082036171369954
# f1_score:  0.751982050772888
# 걸린시간: 693.15 초

#4. RandomOverSampler 적용
# accuracy_score:  0.8173118627498227
# f1_score:  0.7599963457725086
# 걸린시간: 177.56 초