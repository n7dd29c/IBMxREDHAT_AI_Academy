import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = ('./_data/kaggle/bike/bike-sharing-demand/')
train_csv = pd.read_csv(path + ('train.csv'), index_col=0)
# print(train_csv)        # [10886 rows x 11 columns]
test_csv = pd.read_csv(path + ('test.csv'), index_col=0)
# print(test_csv)         # [6493 rows x 8 columns]
submission_csv = pd.read_csv(path + ('sampleSubmission.csv'))
# print(submission_csv)   # [6493 rows x 1 columns]

# print(train_csv.info()) # 결측치 없음
# print(test_csv.info()) # 결측치 없음

x = train_csv.drop(['count', 'casual', 'registered',
                    'season', 'holiday', 'workingday', 'weather'], axis=1)
print(x)                  # [10886 rows x 4 columns]
y = train_csv['count']
print(y.shape)            # (10886,)

train_csv = train_csv.drop(['count', 'casual', 'registered', 
                            'season', 'holiday', 'workingday', 'weather'], axis=1)

test_csv = test_csv.drop(['season', 'holiday', 'workingday', 'weather'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=490
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 5, batch_size = 128)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
rmse = RMSE(y_test, results)

print('loss : ', loss)
print('R2 : ', r2)
print('RMSE : ', rmse)

y_submit = model.predict(test_csv)
# print(y_submit)

submission_csv['count'] = y_submit
# print(submission_csv)

submission_csv.to_csv(path + 'submission_0522_1303.csv')