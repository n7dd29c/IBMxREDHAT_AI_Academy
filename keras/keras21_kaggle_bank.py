# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
import sklearn as sk
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# region
# print(train_csv)
# print(train_csv.head(10))       # train.csv의 제일 앞 10개의 데이터 (default=5)
# print(train_csv.tail(10))       # train.csv의 제일 뒤 10개의 데이터 (default=5)
# print(train_csv.isna().sum())   # 결측치 없음
# print(test_csv.isna().sum())    # 결측치 없음

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited']
# endregion

# 문자 데이터 수치화
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# region
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
# test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
# test_csv['Gender'] = le.fit_transform(test_csv['Gender'])
# print(train_csv['Geography'].value_counts())
# Geography
# 0    94215
# 2    36213
# 1    34606
# print(train_csv['Gender'].value_counts())
# Gender
# 1    93150
# 0    71884
# endregion

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv.shape)  # (165034, 11)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
test_csv = scaler.transform(test_csv)  # 테스트 데이터도 같은 기준으로 변환

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=111,
)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
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
                 validation_split=0.2, callbacks=es, verbose=3)

#4. 평가, 훈련
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc : ', acc_score)

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.plot(hist.history['val_acc'], c='yellow', label='acc')
plt.title('Kaggle Bank')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

y_submit = model.predict(test_csv)
submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_bank_0528.csv')

# 0527_1755
# loss :  0.48298269510269165
# acc :  0.7897718665737571

# 0528_1056
# MinMaxScaler 적용
# loss :  0.3255692422389984
# acc :  0.8620292665192232