import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)          # (7, 3, 1) // (batch_size, timesteps, feature)

# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               [[3],[4],[5]],
#               [[4],[5],[6]],
#               [[5],[6],[7]],
#               [[6],[7],[8]],
#               [[7],[8],[9]]]) 가 된다

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))        # 아래랑 같음
model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # (timesteps, feature) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1) # (3,)를 (1,3,1)로 reshape
y_pred = model.predict(x_pred)
print('11의 예측 : ', y_pred)
