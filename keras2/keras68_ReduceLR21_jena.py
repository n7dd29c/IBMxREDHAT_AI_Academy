import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.optimizers import Adam, Adagrad, SGD, RMSprop # 옵티마이저 클래스 임포트

#1. 데이터

path = './_data/kaggle/jena/'
jena = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
x = np.load(path + 'jena_x_data.npy')
y = np.load(path + 'jena_y_data.npy')
submission_csv = jena['wd (deg)']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[1]*x_train.shape[2]))
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[1]*x_test.shape[2]))

x_train = x_train.reshape(-1, 144, 13)
x_test = x_test.reshape(-1, 144, 13)

#2. 모델구성
model = Sequential()
model.add(GRU(32, input_shape=(144,13), return_sequences=True))
model.add(GRU(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(144))

#3. 컴파일, 훈련
# 옵티마이저 인스턴스를 생성하여 전달합니다.
# === 핵심 수정 부분 ===
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=20, verbose=2, restore_best_weights=True,
)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=2, factor=0.1)
# === 핵심 수정 부분 끝 ===

start_time = time.time()
# verbose=0으로 설정하여 훈련 과정 출력 생략
hist = model.fit(x_train, y_train, epochs= 100, batch_size=128,
                    verbose=2, validation_split=0.1, callbacks=[es,rlr])
end_time = time.time() # time.time()은 함수 호출이 아니므로 괄호 없음 (이전에도 수정됨)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=2) # verbose=0으로 설정하여 평가 과정 출력 생략
results = model.predict(x_test, verbose=2) # verbose=0으로 설정하여 예측 과정 출력 생략
r2 = r2_score(y_test, results)

training_time = end_time - start_time

print(f"Loss: {loss:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training Time: {training_time:.4f} seconds")