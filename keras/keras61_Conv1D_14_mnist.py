import numpy as np
import pandas as pd
import time
import datetime
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# x reshape -> (60000, 28, 28, 1)
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(28,28)))   
model.add(MaxPooling1D())   
model.add(Dropout(0.2))   
model.add(Conv1D(32, 2))   
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))       # input_shape는 생략가능
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

date = datetime.datetime.now().strftime("%y%m%d_%H%M")
path = './_save/keras36_cnn5/'
filename = '{epoch:03d}-{val_loss:.3f}.hdf5'
filepath = ''.join([path, 'k36_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=1,
# )

start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=256, validation_split=0.2,
          callbacks=[es], verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print(end_time - start_time)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = y_test.values
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))

# loss :  0.20601223409175873
# acc :  0.9753999710083008