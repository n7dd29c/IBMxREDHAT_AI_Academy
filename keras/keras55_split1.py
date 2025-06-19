import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

a = np.array(range(1, 11))
timesteps = 5

print(a.shape)  # (10,)
print(len(a))   # 10

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

print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)
x = x.reshape(6,4,1)

x_pred = np.array([7,8,9,10]).reshape(1, 4, 1)

model = Sequential()
model.add(LSTM(512, input_shape=(4,1), return_sequences=True))
model.add(LSTM(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
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
print('11의 예측 : ', y_pred)