import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
np_path = './_data/_save_npy/'
# np.load(np_path + 'keras44_01_x_train.npy', arr=x)
# np.load(np_path + 'keras44_01_y_train.npy', arr=y)

x = np.load(np_path + 'keras46_01_x_train_horse.npy')
y = np.load(np_path + 'keras46_01_y_train_horse.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, 
)

print(x_train.shape, x_test.shape)  # (821, 100, 100, 3) (206, 100, 100, 3)

x_train = x_train.reshape(-1, 300, 100)
x_test = x_test.reshape(-1, 300, 100)
print(x_train.shape, x_test.shape)  # (821, 300, 100) (206, 300, 100)

#2. 모델구성
model = Sequential()
model.add(Conv1D(128, 3, input_shape=(300,100)))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(Conv1D(64, 3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
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
filepath = ''.join([path, 'k46_horse_', filename])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath
)

start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64, callbacks=[es,mcp], validation_split=0.2, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])           # 0.10916756838560104
print(results[1])           # 0.9708737730979919
print(end_time-start_time)  # 72.82964181900024