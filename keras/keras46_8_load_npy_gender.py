import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터

np_path = './_data/_save_npy/'
x = np.load(np_path + 'keras46_01_x_train_gender.npy')
y = np.load(np_path + 'keras46_01_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=55, test_size=0.2, 
)

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