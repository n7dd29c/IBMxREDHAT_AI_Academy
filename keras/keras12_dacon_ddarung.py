# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
# print(np.__version__)   # 1.23.0
# print(pd.__version__)   # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
# print(submission_csv)   # [715 rows x 1 columns]

# region
# print(train_csv.shape)
# print(test_csv.shape)
# print(submission_csv.shape)

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.info())     # 데이터프레임의 구조와 요약정보 (결측치 확인 가능)

# print(train_csv.describe()) # 데이터프레임에 대한 기본적인 통계 요약

######################## 결측치 처리 1. 삭제 ########################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
# print(train_csv.isna().sum())   # isnull() = isna() = info(), 결측치를 확인할 수 있는 함수 

# train_csv = train_csv.dropna()  # 결측치 dropna()로 삭제 후 train_csv에 다시 덮어씌움
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)                # [1328 rows x 10 columns]
# endregion

##################### 결측치 처리 2. 평균치 넣기 ####################
train_csv = train_csv.fillna(train_csv.mean())


#################### 결측치 처리 3. 테스트 데이터 ###################
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)   # drop() : 행 또는 열 삭제
                                        # count라는 열(axis=1) 삭제, 참고로 행은 axis=0
# print(x)                                # [1459 rows x 9 columns]


y = train_csv['count']                  # count 컬럼만 빼서 y에 대입
# print(y.shape)                          # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=888
)

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim = 9, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 800, batch_size = 32, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
print('loss : ',loss)
            
r2 = r2_score(y_test, results)
print('r2: ', r2)   
            
def RMSE(y_test, results):
    return np.sqrt(mean_squared_error(y_test, results))
rmse = RMSE(y_test, results)
print('RMSE : ', rmse) 

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)  # train 데이터의 shape와 동일한 컬럼을 확인하고 넣기
                                    # x_train.shape(N, 9)
# print(y_submit.shape)             # [715 rows x 1 columns]

######### submission.csv 파일 만들기 // count의 컬럼값만 넣어주기 #########
submission_csv['count'] = y_submit
# print(submission_csv)

submission_csv.to_csv(path + 'submission_0522_test.csv')    # csv 만들기


# region ------결과치------

# loss :  2762.022705078125
# r2:  0.5527848034213223
# RMSE :  52.55494914346004

# 1458
# loss :  2302.523193359375
# r2:  0.6379678535556397
# RMSE :  47.98461591392101

# 1502
# loss :  1969.90234375
# r2:  0.6096974202882715
# RMSE :  44.383584007775454

# 1513
# loss :  1877.44091796875
# r2:  0.6280171278508826
# RMSE :  43.32944534504218

# --------------- activation='relu' 활성화 이후 ---------------------

# 1521
# loss :  1683.3206787109375
# r2:  0.6664787338909739
# RMSE :  41.028290647152026

# 1544
# loss :  1562.2000732421875
# r2:  0.6904767322729306
# RMSE :  39.52467512273321

# 1604
# loss :  1256.0987548828125
# r2:  0.7511254854012741
# RMSE :  35.4414817744431

# 

# endregion