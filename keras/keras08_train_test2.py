from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] numpy list 슬라이싱 검색

x_train = x[:7]
y_train = y[:7]
print(x_train)
print(y_train)

x_test = x[7:]
y_test = y[7:]
print(x_test)
print(y_test)

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])



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
model.fit(x_train, y_train, epochs = 200, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print(loss)
print(results)