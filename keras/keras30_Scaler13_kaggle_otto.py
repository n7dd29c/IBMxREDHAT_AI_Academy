import numpy as np
import sklearn as sk
import pandas as pd
import time
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
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

# OneHotEncoder를 사용하여 'target' 문자열을 숫자로 변환
# sparse_output=False (이전에는 sparse=False)를 설정하여 넘파이 배열로 직접 반환
encoder = OneHotEncoder(sparse=False)
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
x_train = scaler.fit_transform(x_train) # x_train에 fit 후 transform
x_test = scaler.transform(x_test)       # x_test에 transform만
test_csv_scaled = scaler.transform(test_csv) # submission을 위한 test_csv에도 transform 적용

# 2. 모델 구성
# Otto 데이터셋의 피처 수는 93개이므로 input_dim=93이 맞습니다.
# 출력 레이어는 One-Hot Encoding된 클래스 수(9개)와 'softmax' 활성화 함수를 사용합니다.
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
    mode='auto',
    patience=30,
    restore_best_weights=True,
)

savepath = './_save/kaggle/otto/' # 저장 경로를 otto에 맞게 변경
date = datetime.datetime.now().strftime('%y%m%d_%H%M')
filename = ('{epoch:03d}-{val_loss:.4f}.hdf5')
filepath = ''.join([savepath, 'otto_', date, '_', filename]) # 파일 이름도 otto에 맞게 변경
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)
start_time = time.time()
model.fit(
    x_train, y_train, epochs=2000, batch_size=512,
    callbacks=[es, mcp], validation_split=0.2, verbose=2
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

# Submission 파일 생성
y_submit_proba = model.predict(test_csv_scaled) # 테스트 데이터에 대한 클래스별 확률 예측

# submission_csv에는 각 클래스에 대한 확률이 필요합니다.
# 컬럼 이름을 'Class_1'부터 'Class_9'까지 맞춰야 합니다.
submission_columns = [f'Class_{i}' for i in range(1, y_encoded.shape[1] + 1)]
submission_df = pd.DataFrame(y_submit_proba, columns=submission_columns, index=test_csv.index)

# 기존 submission_csv의 형태를 유지하며 값만 업데이트
for col in submission_columns:
    submission_csv[col] = submission_df[col]

submission_csv.to_csv(path + 'submission_otto_1.csv')
print('runtime : ', end_time - start_time)
print('submission file is save,', datetime.datetime.now())

# loss :  0.4917704463005066
# acc :  0.8077731132507324
# Log Loss :  0.4917703236650673
# runtime :  35.585965156555176
# submission file is save, 2025-06-09 09:54:09.121420