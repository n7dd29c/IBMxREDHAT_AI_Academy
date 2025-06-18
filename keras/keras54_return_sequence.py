import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(13, 3, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(3,1), activation='relu', return_sequences=True))
model.add(GRU(256, return_sequences=True))
model.add(SimpleRNN(256,))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
    verbose=2
)

path = './_save/keras53/'
filename = '{epoch:04d}.hdf5'
filepath = ''.join([path, 'k53_', filename])
mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
)

model.fit(x, y, epochs=100000, batch_size=2, validation_split=0.2,
          callbacks=[es, mcp], verbose=2)

#4. 평가, 예측
results = model.evaluate(x, y)
print(results[0])

x_pred = x_predict.reshape(1,3,1)
y_pred = model.predict(x_pred)
print(y_pred)