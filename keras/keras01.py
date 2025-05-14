import tensorflow as np
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #optimizer는 adam으로 당분간 고정
model.fit(x, y, epochs=20000)

#4. 평가, 예측
result = model.predict(np.array([6]))
print('6의예측값 : ', result)