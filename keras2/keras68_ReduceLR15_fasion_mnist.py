import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.optimizers import Adam, Adagrad, SGD, RMSprop # 옵티마이저 클래스 임포트

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True))
# print(pd.value_counts(y_test))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv1D(32, 2, input_shape=(28,28)))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(32, 2))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

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
hist = model.fit(x_train, y_train, epochs= 100, batch_size=32,
                    verbose=0, validation_split=0.1, callbacks=[es,rlr])
end_time = time.time() # time.time()은 함수 호출이 아니므로 괄호 없음 (이전에도 수정됨)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=2) # verbose=0으로 설정하여 평가 과정 출력 생략
results = model.predict(x_test, verbose=2) # verbose=0으로 설정하여 예측 과정 출력 생략
r2 = r2_score(y_test, results)

training_time = end_time - start_time

print(f"Loss: {loss:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training Time: {training_time:.4f} seconds")