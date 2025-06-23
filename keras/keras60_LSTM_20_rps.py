import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time

#1. 데이터
np_path = './_data/_save_npy/'

x = np.load(np_path + 'keras46_01_x_train_rps.npy')
y = np.load(np_path + 'keras46_01_y_train_rps.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2,
)

print(x_train.shape, x_test.shape)  # (1638, 100, 100, 3) (410, 100, 100, 3)

x_train, x_test = x_train.reshape(-1, 300, 100), x_test.reshape(-1, 300, 100)
print(x_train.shape, x_test.shape)  # (1638, 300, 100) (410, 300, 100)

#2. 모델구성
model = Sequential()
model.add(LSTM(128, input_shape=(300,100)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

path = './_save/keras46/'
filename = '{epoch:04d}_{val_acc:.4f}.hdf5'
filepath = ''.join([path, 'k46_rps_', filename])
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
print(results[0])   # 0.03738229721784592
print(results[1])   # 0.9902439117431641
print(end_time-start_time)  # 105.11550545692444