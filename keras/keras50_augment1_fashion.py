from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

datagen = ImageDataGenerator(   # 인스턴스화, 실행 할 준비
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,      # 수치 평행이동 10%
    height_shift_range=0.1,
    zoom_range=1.2,
    rotation_range=10,          # 5도 회전
    fill_mode='nearest',        # 빈 공간 근접치로 채움
)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)                          # 
print(np.min(randidx), np.max(randidx)) # 

x_augmented = x_train[randidx].copy()   # 새로운 메모리 공간에 x_augment 할당, 메모리 병합 방지
                                        # x_augment와 copy는 서로 영향을 주지 않음

y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape)                # 
print(y_augmented.shape)                # 

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],               #
    x_augmented.shape[1],               # 
    x_augmented.shape[2],               #
    1,
    )
print(x_augmented.shape)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,                              # x만 있는 상태에선 shuffle 위험  
).next()[0]

print(x_augmented.shape)                        #

print(x_train.shape)                            
x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)              #

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)             #

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(16, 2, 1, input_shape=(28, 28, 1)))    # filters, kernel_size, strides 생략
model.add(Conv2D(8, 3, 1))
model.add(MaxPooling2D())
model.add(Conv2D(8, 3, 1))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=256, validation_split=0.2,
          callbacks=[es], verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print(end_time-start_time)

# augment 전
# loss :  0.45550528168678284
# acc :  0.8575000166893005
# 147.2709023952484

# augment 후
# loss :  0.40302783250808716
# acc :  0.8601999878883362
# 120.64219689369202