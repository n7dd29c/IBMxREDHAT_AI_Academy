# train.csv와 new_test.csv로 count 예측

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
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)    # [10886 rows x 11 columns]
new_test_csv = pd.read_csv(path + 'new_test.csv')
# print(new_test_csv) # [6493 rows x 10 columns]
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
# print(submission_csv) # [6493 rows x 2 columns]

x = train_csv.drop(['count'], axis=1)   # train_csv의 count열을 제거
# print(x)          # [10886 rows x 10 columns]
y = train_csv['count']                  # train_csv의 count열만 적용
# print(y.shape)    # (10886,) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=111
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size=32) 
# model에 학습 데이터 저장
# model은 학습이 된 상태고 바로 예측에 사용할 수 있다

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)          # r2_score(y의 정답값, y의 예측값)

def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))
rmse = RMSE(y_test, results)

print(f"loss : {loss}\nR2 : {r2}\nRMSE : {rmse}")

y_submit = model.predict(new_test_csv)  # model에 저장된 학습데이터로 new_test_csv를 예측
submission_csv['count'] = y_submit      # submission_csv의 count열에 y_submit 대입

submission_csv.to_csv(path + 'submission_test.csv', index=False)
# loss : 0.004317997023463249
# R2 : 0.9999998752511035
# RMSE : 0.06571023410562683