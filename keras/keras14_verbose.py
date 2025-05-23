# keras08-1 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)
# print(y.shape)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 1, verbose=3)
# verbose = -1 : epoch 진행상황 출력
# verbose = 0 : 침묵
# verbose = 1 : default
# verbose = 2 : progress bar 삭제
# verbose = 3 : -1과 동일

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print(loss)
print(results)