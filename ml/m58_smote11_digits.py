# smote는 증폭을 통해 보간처리 방식

import numpy as np
import pandas as pd
import random
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization   
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=seed,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

############### SMOTE ###############
#꼭 split 하고 SMOTE 쓰기
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version: ', sk.__version__) #1.6.1
import imblearn
print('imblearn version: ', imblearn.__version__) #0.12.4

ros = RandomOverSampler(random_state=seed,
                        sampling_strategy={0:2000, 1:2000, 2:2000, 3:2000, 4:2000, 5:2000, 6:2000, 7:2000, 8:2000, 9:2000}  #직접 지정
                        )
x_train , y_train = ros.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([267, 267], dtype=int64))

#2. 모델
model = RandomForestClassifier(random_state=seed)

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', #one-hot 을 안 했기 때문에 categorical_crossentropy가 아님
#               optimizer = 'adam', metrics=['acc']) 
# es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print('accuracy_score:', acc)
print('f1_score:', f1)
print("걸린시간:", round(end - start, 2), '초')

############### 결과 ###############
#1. 변환하지 않은 원 데이터 훈련
# accuracy_score: 0.9822222222222222
# f1_score: 0.9820950090752157

#2. 2번에서 SMOTE 적용
# accuracy_score: 0.9844444444444445
# f1_score: 0.9843146327356853

#3. 재현's SMOTE
# accuracy_score: 0.9866666666666667
# f1_score: 0.9866331787134058

# RandomOverSampler
# accuracy_score: 0.9866666666666667
# f1_score: 0.986609015561613