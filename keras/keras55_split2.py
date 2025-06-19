import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T
print(a)

timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps=timesteps)
print(bbb)

x = bbb[:,:-1]
y = bbb[:,-1]

print(x, x.shape)   # (6, 4, 2)
print(y, y.shape)   # (6, 2)

x_pred = np.array([[7,3],[8,2],[9,1],[10,0]]).reshape(1,4,2)

model = Sequential()
model.add(LSTM(512, input_shape=(4,2), return_sequences=True))
model.add(LSTM(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)
model.fit(x, y, epochs=10000, batch_size=2, validation_split=0.2, callbacks=es, verbose=2)

results = model.evaluate(x,y)
print(results[0])

y_pred = model.predict(x_pred)
print('[11, -1]의 예측 : ', y_pred)