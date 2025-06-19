import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

a = np.array(range(1,101))
x_pred = np.array(range(96,106))

timesteps = 6

print(a, x_pred)

def x_split(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = x_split(a, timesteps=timesteps)
x_predict = x_split(x_pred, timesteps=timesteps-1)

print(bbb)
print(bbb.shape)    # (95, 6)

x = bbb[:,:-1]
y = bbb[:,-1]

print(x, y)
print(x.shape, y.shape) # (95, 5) (95,)
print(x_predict.shape)  # (6, 5)

x = x.reshape(-1, 5, 1)
x_predict = x_predict.reshape(-1, 5, 1)

model = Sequential()
model.add(LSTM(256, input_shape=(5,1), return_sequences=True))
model.add(LSTM(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.fit(x, y, epochs=10000, batch_size=32, validation_split=0.2, verbose=2, callbacks=es,)

results = model.evaluate(x,y)
print(results[0])
y_predict = model.predict(x_predict)
print('100 ~ 106 예측 : ', y_predict)

# 0.8948748111724854
# 106 예측 :  [100.93798]