import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#1. 데이터

np_path = './_data/_save_npy/'
x = np.load(np_path + 'keras46_01_x_train_gender.npy')
y = np.load(np_path + 'keras46_01_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=55, test_size=0.2, 
)

datagen = ImageDataGenerator(   # 인스턴스화, 실행 할 준비
    rescale=1./255,             # 0~255 정규화 (scaling)
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,      # 수치 평행이동 10%
    height_shift_range=0.1,
    zoom_range=1.2,
    rotation_range=10,          # 5도 회전
    fill_mode='nearest',        # 빈 공간 근접치로 채움
)

augment_size = 10000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# np.random.randint(60000, 40000)       # 위에 거랑 같은 결과
print(randidx)                          # [32476 53266 14516 ... 51742 39369 37640]
print(np.min(randidx), np.max(randidx)) # 1 59998

x_augmented = x_train[randidx].copy()   # 새로운 메모리 공간에 x_augment 할당, 메모리 병합 방지
                                        # x_augment와 copy는 서로 영향을 주지 않음

y_augmented = y_train[randidx].copy()

print(x_augmented)                      
print(x_augmented.shape)                # (10000, 100, 100, 3)
print(y_augmented.shape)                # (10000,)

# x_augmented = x_augmented.reshape(40000, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],               # 40000
    x_augmented.shape[1],               # 28
    x_augmented.shape[2],               # 28
    3,
    )
print(x_augmented.shape)                # (10000, 100, 100, 3)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,                              # x만 있는 상태에선 shuffle 위험  
    save_to_dir='c://study25//_data//_save_img//08_gender//',
).next()[0]

print(x_augmented.shape)                        # (10000, 100, 100, 3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)             # (12647, 100, 100, 3) (12647,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, 3, 1, input_shape=(100,100,3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(256, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
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
    patience=30,
    restore_best_weights=True,
) 

path = './_save/keras46/'
filename = '{epoch:04d}_{val_acc:.4f}.hdf5'
filepath = ''.join([path, 'k46_gender_', filename])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath
)

start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64, callbacks=[es,mcp],
          validation_split=0.2, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])           # 0.6519955396652222
print(results[1])           # 0.6253776550292969
print(end_time-start_time)  # 108.46539497375488

# k46_gender_0001_0.5962
# 0.6679198145866394
# 0.6057401895523071
# 186.28261494636536

# k46_gender_0010_0.6283
# 0.6617336869239807
# 0.638972818851471
# 182.86690497398376

# 0.686881422996521
# 0.555891215801239
# 632.3908751010895