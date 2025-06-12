# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
import sklearn as sk
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

#1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자 데이터 수치화
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv.shape)  # (165034, 11)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=111,
)

print(x_train.shape, y_train.shape) # (132027, 10) (132027,)
print(x_test.shape, y_test.shape)   # (33007, 10) (33007,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)  # 테스트 데이터도 같은 기준으로 변환

x_train = x_train.reshape(-1, 5, 2, 1)
x_test = x_test.reshape(-1, 5, 2, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, 2, 1, input_shape=(5,2,1), padding='same'))
model.add(Conv2D(32, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=512,
                 validation_split=0.2, callbacks=[es], verbose=3)

#4. 평가, 훈련
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc : ', acc_score)

# StandardScaler
# loss :  0.32277700304985046
# acc :  0.8638773593480171

# dropout
# loss :  0.34481820464134216
# acc :  0.8579089284091254

# loss :  0.32090944051742554
# acc :  0.8645741812342836