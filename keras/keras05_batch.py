from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터정의
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#epoch를 100으로 고정
#loss 기준 0.32미만으로 만들기

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 3)

#4. 예측, 평가
loss = model.evaluate(x,y)
print('################################################################')
print('epochs : ', epochs)
print('loss : ', loss)
# results = model.predict([7])
# print('6의 예측값 : ', results)

# epochs :  100
# loss :  0.32392629981040955