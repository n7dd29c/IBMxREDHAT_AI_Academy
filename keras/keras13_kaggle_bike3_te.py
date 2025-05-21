import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = ('./_data/kaggle/bike/bike-sharing-demand/') # 상대경로
# path = ('.\_data\kaggle\bike\bike-sharing-demand/') # \n \a \b 등 예약어를 제외하면 가능
# path = ('.//_data//kaggle//bike//bike-sharing-demand/')
# path = ('.\\_data\\kaggle\\bike\\bike-sharing-demand/')
# path = 'C:/Study25/_data/kaggle/bike/bike-sharing-demand/' # 절대경로
# / 와 \ 를 섞어서 쓸 수도 있다

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

'''
#region 데이터 정보
print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission_csv.shape) # (6493, 2)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']


print(test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed']

print(submission_csv.columns)
# Index(['datetime', 'count']

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())    # 결측치 없음

print(train_csv.describe())     # 데이터의 통계
#endregion
'''

x = train_csv.drop(['count'], axis=1)
print(x)        # [10886 rows x 10 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=853
)

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))    # activation='linear'는 default값이라 생략가능

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1, batch_size = 64)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('R2 : ', r2)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
rmse = RMSE(y_test, results)
print('RMSE : ', rmse)

y_submit = model.predict(new_test_csv)
# print(y_submit)

submission_csv['count'] = y_submit
# print(submission_csv)

print(new_test_csv)
submission_csv.to_csv(path + 'submission_0523_teacher.csv', index=False)

# loss :  21546.39453125
# R2 :  0.30183010892581164
# RMSE :  146.78690841928383