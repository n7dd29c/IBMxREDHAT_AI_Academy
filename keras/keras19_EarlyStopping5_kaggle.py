import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd
import time
from tensorflow.python.keras.callbacks import EarlyStopping
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
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))    # activation='linear'는 default값이라 생략가능

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs = 1000, batch_size = 128,
          validation_split=0.2, callbacks=es)
end_time = time.time()


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
print('Time : ', end_time - start_time)

y_submit = model.predict(test_csv)
# print(y_submit)

submission_csv['count'] = y_submit
# print(submission_csv)

submission_csv.to_csv(path + 'submission_ES.csv', index=False)

# region 결과값

# loss :  21546.39453125
# R2 :  0.30183010892581164
# RMSE :  146.78690841928383

# validation 사용 후
# loss :  25658.009765625
# R2 :  0.16860116803924274
# RMSE :  160.18117340420739

# EarlyStopping
# loss :  25892.646484375
# R2 :  0.16099844064932756
# RMSE :  160.9118951288683

# loss :  24140.060546875
# R2 :  0.27975047610462045
# RMSE :  155.37072826524533

# loss :  23317.646484375
# R2 :  0.30428846926879716
# RMSE :  152.70115155099242

# loss :  22613.302734375
# R2 :  0.33162494581143265
# RMSE :  150.37719156475038

# endregion