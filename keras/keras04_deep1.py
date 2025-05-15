from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터정의
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100, input_dim=100))    
model.add(Dense(100, input_dim=100))
model.add(Dense(100, input_dim=100))
model.add(Dense(100, input_dim=100))
model.add(Dense(100, input_dim=100))
model.add(Dense(99, input_dim=100))
model.add(Dense(1, input_dim=99))

epochs = 300

#3. 컴파일,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=epochs)

#4. 평가,예측
loss = model.evaluate(x,y)
print('############################################################################')
print('epochs : ', epochs)
print('loss : ', loss)
results = model.predict([6])
print('6의 예측값', results)

# epochs :  300
# loss :  2.877698079828406e-13
# 1/1 [==============================] - 0s 86ms/step
# 6의 예측값 [[6.0000014]]