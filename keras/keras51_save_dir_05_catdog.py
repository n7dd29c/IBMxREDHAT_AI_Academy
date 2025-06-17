import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
submit_path = './_data/kaggle/cat_dog/'
x = np.load(np_path + 'keras44_01_x_train_catdog.npy')
y = np.load(np_path + 'keras44_01_y_train_catdog.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, 
)
print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)

agu_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    # 좌우 반전
    vertical_flip=True,      # 상하 반전
    # width_shift_range=0.1,   # 가로로 10% 이동
    # height_shift_range=0.1,  # 세로로 10% 이동
    # zoom_range=1.2,          # 최대 20% 확대
    # rotation_range=10,       # 최대 10도 회전
    # fill_mode='nearest',     # 빈 공간은 주변 색으로 채움  
)

augment_size=1000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = agu_data_gen.flow(
    x_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='c://study25//_data//_save_img//05_catdog//',
).next()

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (25000, 100, 100, 3) (25000,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, 3, 1, input_shape=(100,100,3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(256, 3, 1, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

path = './_save/keras45/'
filename = '{epoch:04d}_{val_acc:.4f}.hdf5'
filepath = ''.join([path, 'k45_catdog_', filename])
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
print(results[0])           # 0.5785045623779297
print(results[1])           # 0.7361999750137329
print(end_time-start_time)  # 845.2242391109467

# 0.5124953985214233
# 0.7495999932289124
# 3199.773459672928