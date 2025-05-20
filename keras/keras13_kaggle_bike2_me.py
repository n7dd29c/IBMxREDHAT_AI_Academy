#1. train_csv 에서 casual registered 를 y로 잡는다
#2. 훈련해서, test_csv의 casual과 registered를 예측(predict)한다
#3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다
#   (N, 8) -> (N, 10)   test.csv 파일로 new_test.csv 파일을 만든다

import numpy as np
import sklearn as sk
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = ('./_data/kaggle/bike/bike-sharing-demand/')
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
train_csv = train_csv.drop(['count'], axis=1)   # train.csv의 count열을 삭제
print(train_csv)    # [10886 rows x 10 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [6493 rows x 8 columns]

x = train_csv.drop(['casual', 'registered'], axis=1)
                    # train.csv의 casual, registered열을 삭제 후 x로 선언
# print(x)    # [10886 rows x 8 columns]
y = train_csv[['casual', 'registered']]
                # 이미 casual과 registered가 벡터형태이기때문에 []를 씌워 행렬로 만듬
                # train.csv의 casual과 registered열만 빼서 y로 선언
# print(y)    # [10886 rows x 2 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=146
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=32)

#4. 평가, 예측
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

print('loss : ', loss)
print('R2 : ', r2)
print('RMSE : ', rmse)

new_test = model.predict(test_csv)
test_csv[['casual', 'registered']] = new_test

test_csv.to_csv(path + 'new_test.csv', index=False)

# loss :  10858.0439453125
# R2 :  0.20278439519799935
# RMSE :  104.20193575847564