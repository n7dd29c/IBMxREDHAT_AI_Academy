#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터는 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개이상 100개이하
#6. train_size=0.75
#7. epochs>=100
#8. loss = mse
#9. dropout 금지

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=333
)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))         
model.add(Dense(90, activation='relu'))   
model.add(Dense(80, activation='relu'))   
model.add(Dense(70, activation='relu'))  
model.add(Dense(60, activation='relu'))  
model.add(Dense(50, activation='relu'))  
model.add(Dense(40, activation='sigmoid')) 
model.add(Dense(1))  

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'sgd', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict(x_test)
print(loss[1])
print(result)
# 0.05000000074505806