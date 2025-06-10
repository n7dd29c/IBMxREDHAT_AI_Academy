from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# (N,5,5,1) 이미지, N은 데이터의 갯수

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))
#                                                            ↑ ↑ ↑
#                                                            | | |
#                                                            | | channels (색)
#                                                            | width (가로)
#                                                            height (세로)
model.add(Conv2D(5, (2,2))) 
model.add(Flatten())       # 데이터의 내용과 순서는 바꾸지 않고 (1, N)으로 바꿔줌
model.add(Dense(10))       # Dense는 다차원 입력이 가능, 출력도 무조건 다차원
model.add(Dense(units=3))  # input = (batch_size, input_dim)

model.summary()