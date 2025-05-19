from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    # train_size=0.7,  # 생략가능, default = 0.75
                                    test_size=0.3,     # default = 0.25
                                    shuffle=True,      # default = True
                                    random_state=121)     

print(x_train)
print(x_test)
print(y_train)
print(y_test)


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