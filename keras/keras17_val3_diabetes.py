from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# [실습] r2 = 0.62

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x)    # (442, 10)
# print(y)    # (442,)
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=345
)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim = 10))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

r2 = r2_score(y_test, results)
print(r2)
# 0.6206238309504791 <- validation 이전
# 0.5034337284438537