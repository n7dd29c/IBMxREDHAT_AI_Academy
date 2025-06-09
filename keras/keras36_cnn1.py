from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# (N,5,5,1) 이미지, N은 데이터의 갯수

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))   # (None, 4, 4, 10)
model.add(Conv2D(5, (2,2)))                         # (None, 3, 3,  5)

model.summary()