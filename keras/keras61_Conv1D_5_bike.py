import pandas as pd
import time
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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

print(x_train.shape, y_train.shape) # (8708, 8) (8708,)
print(x_test.shape, y_test.shape)   # (2178, 8) (2178,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(8,1)))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))    # activation='linear'는 default값이라 생략가능

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=10, restore_best_weights=True
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 256,
          verbose=2, validation_split=0.2, callbacks=es)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)
print(round(end_time-start_time, 2))

# dropout
# 0.2111162027142952

# 0.2870492064567006