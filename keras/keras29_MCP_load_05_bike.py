import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential, load_model
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
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

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

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=7572
)

#2. 모델구성
path = './_save/keras28_mcp/05_bike/'
model = load_model(path + 'k28250604_1212_0147-21605.1367.hdf5')

#3. 컴파일, 훈련

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

# loss :  22420.13671875
# R2 :  0.28240844318873937
# RMSE :  149.73355697525346

# loss :  22420.13671875
# R2 :  0.28240844318873937
# RMSE :  149.73355697525346