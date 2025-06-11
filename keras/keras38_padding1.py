# 실습
# (100, 100, 3) 이미지를 (10, 10, 11)으로 줄이기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(11, 2, 2, input_shape=(100, 100, 3)))
model.add(Conv2D(11, 3, strides=2, padding='valid'))
model.add(MaxPooling2D())
model.add(Conv2D(11, 3, strides=1, padding='valid'))
model.summary()

#  conv2d_4 (Conv2D)           (None, 10, 10, 11)        14652