from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 패션 MNIST 데이터 불러오기

x_train = x_train/255.   # 0~1로 정규화
x_test = x_test/255.     # 0~1로 정규화

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# 데이터 증강 설정: 여러 방법으로 이미지 변형
datagen = ImageDataGenerator(
    horizontal_flip=True,    # 좌우 반전
    vertical_flip=True,      # 상하 반전
    width_shift_range=0.1,   # 가로로 10% 이동
    height_shift_range=0.1,  # 세로로 10% 이동
    zoom_range=1.2,          # 최대 20% 확대
    rotation_range=10,       # 최대 10도 회전
    fill_mode='nearest',     # 빈 공간은 주변 색으로 채움
)

augment_size = 50000  # 증강할 데이터 개수

# 원본 데이터에서 랜덤으로 40000개 샘플 선택
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)                           # 선택된 인덱스 출력
print(np.min(randidx), np.max(randidx))  # 최소, 최대 인덱스 출력

# 선택된 샘플 복사해서 새 배열 만들기 (원본 손상 방지)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape)  # 
print(y_augmented.shape)  # 

# CNN에 맞게 차원 추가 (채널=1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],  # 
    x_augmented.shape[1],  # 
    x_augmented.shape[2],  # 
    3,                     # 채널
)
print(x_augmented.shape)   #

# 설정한 datagen으로 증강 데이터 생성 (이미지 배치만 사용)
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,           # 라벨도 같이 주지만 아래에서 [0]로 이미지만 받음
    batch_size=augment_size,
    shuffle=False,
).next()[0]                # 이미지 배치만 가져오기

print(x_augmented.shape)   #

# 원본 훈련 데이터 모양도 CNN에 맞게 reshape
print(x_train.shape)       # 
x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 3)
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 3)
print(x_train.shape, x_test.shape)  # 

# 원본 + 증강 데이터를 합쳐서 새로운 훈련 세트 만들기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)

# 라벨을 원-핫 인코딩으로 변환
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(128, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D()) 
model.add(Conv2D(100, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])
print(results[1])
print(end_time-start_time)

# 1.3969573974609375
# 0.5185999870300293
# 260.4936192035675

# 0.9577142596244812
# 0.6664999723434448
# 408.9307882785797