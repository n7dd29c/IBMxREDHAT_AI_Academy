import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,0]])
x = np.transpose(x) # x = np.array([[1,6],[2,7],[3,8],[4,9],[5,0]])과 같은 효과

y = np.array([1,2,3,4,5])

print(x.shape) # (5, 2)
print(y.shape) # (5, )

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=2))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([[6,11]]) # (1, 2)
print('loss : ', loss)
print('[6,11]의 예측값 : ', results)