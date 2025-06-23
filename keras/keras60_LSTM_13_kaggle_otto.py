import numpy as np
import sklearn as sk
import pandas as pd
import time
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss # Otto는 Multi-class Log Loss를 평가 지표로 사용
from sklearn.preprocessing import StandardScaler, OneHotEncoder # OneHotEncoder 추가

# 1. 데이터 로드 및 전처리
path = './_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(f"Original train_csv shape: {train_csv.shape}") # (61878, 94) - Otto 데이터셋 기준
print(f"Original test_csv shape: {test_csv.shape}")   # (144368, 93) - Otto 데이터셋 기준

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# --- Otto 데이터셋의 'target' 컬럼 처리 (One-Hot Encoding) ---
print(f"Initial y dtype: {y.dtype}")
print(f"Initial y unique values and counts:\n{y.value_counts().sort_index()}")

# train/test split
# Otto는 다중 클래스 분류이므로 y_encoded를 사용합니다.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y # stratify는 원본 y 사용
)

print(f"x_train shape: {x_train.shape}")    # (49502, 93)
print(f"y_train shape: {y_train.shape}")    # (49502,)
print(f"x_test shape: {x_test.shape}")      # (12376, 93)
print(f"y_test shape: {y_test.shape}")      # (12376,)

# --- 스케일러 적용 ---
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # x_train에 fit 후 transform
x_test = scaler.transform(x_test)       # x_test에 transform만
test_csv_scaled = scaler.transform(test_csv) # submission을 위한 test_csv에도 transform 적용

x_train = x_train.reshape(-1, 31, 3, 1)
x_test = x_test.reshape(-1, 31, 3, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.values.reshape(-1, 1))
y_test = ohe.transform(y_test.values.reshape(-1, 1))
print(y_train.shape, y_test.shape)  # (49502, 9) (12376, 9)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, 2, 1, input_shape=(31,3,1), padding='same'))
model.add(Conv2D(32, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))

# 3. 컴파일, 훈련
# 다중 클래스 분류이므로 'categorical_crossentropy'를 사용합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train, epochs=2000, batch_size=512,
    callbacks=[es], validation_split=0.2, verbose=2 
)
end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', results[0])
print('acc : ', results[1])

# 다중 클래스 예측은 log_loss로 평가합니다.
y_predict = model.predict(x_test)
# log_loss는 예측된 확률과 실제 원-핫 인코딩된 레이블을 사용합니다.
logloss = log_loss(y_test, y_predict)
print('Log Loss : ', logloss)
print('runtime : ', end_time - start_time)

# dropout
# loss :  0.4917704463005066
# acc :  0.8077731132507324
# Log Loss :  0.4917703236650673
# runtime :  35.585965156555176

# loss :  0.5405053496360779
# acc :  0.7962185144424438
# Log Loss :  0.5405052257933523
# runtime :  21.756399869918823