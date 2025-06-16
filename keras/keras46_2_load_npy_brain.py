import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
# np.load(np_path + 'keras44_01_x_train.npy', arr=x)
# np.load(np_path + 'keras44_01_y_train.npy', arr=y)

x = np.load(np_path + 'keras46_01_x_train_brain.npy')
y = np.load(np_path + 'keras46_01_y_train_brain.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=55, test_size=0.2,
)

#2. 모델구성
model = Sequential()
model.add(Conv2D(100, 3, 1, input_shape=(100,100,1)))
model.add(Conv2D(128, 3, 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(128, 3, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

path = './_save/keras46/'
filename = '{epoch:04d}_{val_acc:.4f}.hdf5'
filepath = ''.join([path, 'k46_brain_', filename])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath
)

start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64,
          callbacks=[es,mcp], validation_split=0.2, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])                        # 0.08176270127296448
print(results[1])                        # 0.96875
print(np.round(end_time-start_time, 2))  # 35.87