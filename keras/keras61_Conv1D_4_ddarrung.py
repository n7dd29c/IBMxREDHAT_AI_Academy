# https://dacon.io/competitions/open/235576/overview/description
import time
import numpy as np
import pandas as pd
# print(np.__version__)   # 1.23.0
# print(pd.__version__)   # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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
    test_size=0.2,
    random_state=444
)

print(x_train.shape, y_train.shape) # (1167, 9) (1167,)
print(x_test.shape, y_test.shape)   # (292, 9) (292,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 9, 1)
x_test = x_test.reshape(-1, 9, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(9,1)))
model.add(MaxPooling1D())
model.add(Conv1D(32, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=10, restore_best_weights=True
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
          verbose=1, validation_split=0.1, callbacks=es)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)

# 0.6375322985925097