import numpy as np
import sklearn as sk
import pandas as pd
import time
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss # Otto는 Multi-class Log Loss를 평가 지표로 사용
from sklearn.preprocessing import StandardScaler, OneHotEncoder # OneHotEncoder 추가
from imblearn.over_sampling import SMOTE, SMOTENC

# 1. 데이터 로드 및 전처리
path = './Study25/_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.info())

print(f"Original train_csv shape: {train_csv.shape}") # (61878, 94) - Otto 데이터셋 기준
print(f"Original test_csv shape: {test_csv.shape}")   # (144368, 93) - Otto 데이터셋 기준

x = train_csv.drop(['target'], axis=1)  # (61878, 94)
y = train_csv['target']                 # (61878,)


# --- Otto 데이터셋의 'target' 컬럼 처리 (One-Hot Encoding) ---
print(f"Initial y dtype: {y.dtype}")
print(f"Initial y unique values and counts:\n{y.value_counts().sort_index()}")

# OneHotEncoder를 사용하여 'target' 문자열을 숫자로 변환
# sparse_output=False (이전에는 sparse=False)를 설정하여 넘파이 배열로 직접 반환
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

print(f"Shape of y after One-Hot Encoding: {y_encoded.shape}")
print(f"First 5 rows of y_encoded:\n{y_encoded[:5]}")

# train/test split
# Otto는 다중 클래스 분류이므로 y_encoded를 사용합니다.
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=55, stratify=y # stratify는 원본 y 사용
)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 스케일러 적용 ---
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)      # x_train에 fit 후 transform
x_test = scaler.transform(x_test)            # x_test에 transform만
test_csv_scaled = scaler.transform(test_csv) # submission을 위한 test_csv에도 transform 적용

# smote = SMOTE(random_state=111)
# x_res, y_res = smote.fit_resample(x_train, y_train)

smotenc = SMOTENC(random_state=111, categorical_features=[11,22,33,44,55])
x_res, y_res = smotenc.fit_resample(x_train, y_train)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_encoded.shape[1], activation='softmax')) # 출력 레이어 변경: 클래스 수, softmax

# 3. 컴파일, 훈련
# 다중 클래스 분류이므로 'categorical_crossentropy'를 사용합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=2
)

start_time = time.time()
model.fit(
    x_res, y_res, epochs=2000, batch_size=512,
    callbacks=es, validation_split=0.2, verbose=2
)
end_time = time.time() - start_time

# 4. 평가, 예측
results = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', results[0])
print('acc : ', results[1])

# 다중 클래스 예측은 log_loss로 평가합니다.
y_predict = model.predict(x_test)
# log_loss는 예측된 확률과 실제 원-핫 인코딩된 레이블을 사용합니다.
logloss = log_loss(y_test, y_predict)
print('Log Loss : ', logloss)
print(f'{int(end_time)}초')

# 원값
# loss :  0.4917704463005066
# acc :  0.8077731132507324
# Log Loss :  0.4917703236650673
# 35초

# SMOTE 적용
# loss :  0.5699827671051025
# acc :  0.7908855676651001
# Log Loss :  0.5693809386947657
# 130초

# SMOTENC 적용
# loss :  0.5714854001998901
# acc :  0.7925016283988953
# Log Loss :  0.5705453436501504
# 146초