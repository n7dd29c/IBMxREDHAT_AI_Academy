# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTE

#1. 데이터
path = './Study25/_data/kaggle/bank/'
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

print(train_csv.info())

# le_geo.fit(train_csv['Geography'])  # fit()은 train만!
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

# le_gen.fit(train_csv['Gender'])     # fit()은 train만!
# train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
# test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# region
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])
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

# SMOTE
smote = SMOTE(random_state=111)
x_res, y_res = smote.fit_resample(x_train, y_train)

# SMOTENC
# smotenc = SMOTENC(random_state=111, categorical_features=[2,3,6,7,8],)
# x_res, y_res = smotenc.fit_resample(x_train, y_train)

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
start = time.time()
model.fit(x_res, y_res, epochs=1000, batch_size=512,
          validation_split=0.2, callbacks=es, verbose=2)
end = time.time() - start

#4. 평가, 훈련
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc : ', acc_score)
print(f'걸린시간 : {int(end)}초')

# 원값
# loss :  0.3255692422389984
# acc :  0.8620292665192232

# SMOTE 적용
# loss :  0.4000549018383026
# acc :  0.8260671978671191

# SMOTENC 적용
# loss :  0.43215230107307434
# acc :  0.797558093737692