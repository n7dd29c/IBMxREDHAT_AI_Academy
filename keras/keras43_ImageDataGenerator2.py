import time
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 0~255 정규화 (scaling)
    # horizontal_flip=True,   # 수평 뒤집기, 데이터 증폭 (변환)
    # vertical_flip=True,     # 수직 뒤집기, 데이터 증폭 (변환)
    # width_shift_range=0.1,  # 수치 평행이동 10%
    # height_shift_range=0.1, # 수치 수직이동 10%
    # rotation_range=5,       # 5도 회전
    # zoom_range=1.2,         # 1.2배 확대
    # shear_range=0.7,        # 좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시켜 변환 (이미지 늘리기)
    # fill_mode='nearest',    # 빈 공간 근접치로 채움
)

test_datagen = ImageDataGenerator(  # 검증데이터는 증폭, 변환하면 안됨
    rescale=1./255  
)

train_path = './_data/image/brain/train/'
test_path = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    train_path,             # 경로
    target_size=(100, 100), # 사이즈 규격 일치 (큰거는 축소, 작은건 확대)
    batch_size=160,
    class_mode='binary',    # 분류
    color_mode='grayscale',
    shuffle=True,           # default=False    
)

xy_test = test_datagen.flow_from_directory(
    test_path,              # 경로
    target_size=(100, 100), # 사이즈 규격 일치 (큰거는 축소, 작은건 확대)
    batch_size=160,
    class_mode='binary',    # 분류
    color_mode='grayscale',
    # shuffle=True,         # test 데이터는 섞지 않음    
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
print(x_train.shape, y_train.shape) # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 200, 200, 1) (120,)


#2. 모델구성
model = Sequential()
model.add(Conv2D(512, 2, 1, input_shape=(100, 100, 1)))
model.add(Conv2D(256, 2, 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(128, 2, 1, activation='relu'))
model.add(Conv2D(64, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train, epochs=3000, validation_split=0.2,
          callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print('runtime : ', end_time-start_time)

import matplotlib.pyplot as plt
plt.imshow(x_test[1], 'gray')
plt.show()

# loss :  0.10088679194450378
# acc :  0.949999988079071
# runtime :  47.03994417190552