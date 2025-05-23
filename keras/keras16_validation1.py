# keras08-1 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6])   # 훈련데이터
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])             # 검증데이터
y_val = np.array([7,8])

x_test = np.array([9,10])           # 평가데이터
y_test = np.array([9,10])


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
model.fit(x_train, y_train, epochs = 200, batch_size = 1,
          validation_data=(x_val, y_val)
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print(loss)
print(results)