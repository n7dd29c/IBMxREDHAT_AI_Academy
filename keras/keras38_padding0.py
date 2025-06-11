from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10, 10, 1),
                 strides=1, padding='same'))   # padding='valid'가 default
model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=1, padding='valid'))
model.add(Conv2D(8, 4)) # filters 생략, kernel_size 생략 (자동으로 4,4 인식)
model.summary()