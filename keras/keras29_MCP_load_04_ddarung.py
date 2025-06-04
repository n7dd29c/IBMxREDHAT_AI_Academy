# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
# print(np.__version__)   # 1.23.0
# print(pd.__version__)   # 2.2.3

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    test_size=0.1,
    random_state=444
)

#2. 모델구성

#3. 컴파일, 훈련
path = './_save/keras28_mcp/04_ddarung/'
model = load_model(path + 'k28250604_1204_0060-2665.0500.hdf5')

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

# loss :  1968.362548828125
# r2:  0.6584614281579446
# RMSE :  44.3662285993815

# loss :  1968.362548828125
# r2:  0.6584614281579446
# RMSE :  44.3662285993815