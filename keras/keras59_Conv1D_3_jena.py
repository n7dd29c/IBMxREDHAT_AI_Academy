# jena_안태영_submit.csv
# scv = datetime, wd
# RNN 계열 모델 사용
# 16.12.31 00:10 부터
# 17.01.01 00:00 까지 예측
# RMSE

import numpy as np
import pandas as pd
import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

path = './_data/kaggle/jena/'
jena = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
x = np.load(path + 'jena_x_data.npy')
y = np.load(path + 'jena_y_data.npy')
submission_csv = jena['wd (deg)']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

print(x_train.shape, y_train.shape) # (2335, 144, 13) (2335, 144)
print(x_test.shape, y_test.shape)   # (584, 144, 13) (584, 144)

x_train = x_train.reshape(-1, 144*13)
x_test = x_test.reshape(-1, 144*13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # -6329.449367088608 15.354609929078014
print(np.min(x_test), np.max(x_test))   # -4.290821256038635 15.191489361702127

x_train = x_train.reshape(-1, 144, 13)
x_test = x_test.reshape(-1, 144, 13)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=2, input_shape=(144,13), padding='same'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(256, 3))
model.add(MaxPooling1D())
model.add(Conv1D(128, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(144))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

date = datetime.datetime.now().strftime('%m%d_%M%S')
filename = ('{epoch:04d}_{val_loss:.4f}.hdf5')
filepath = ''.join([path, 'jena_', date, '_', filename])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath
)
model.fit(x, y, epochs=10000, batch_size=64, validation_split=0.2, verbose=2, callbacks=[es,mcp],)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])
y_pred = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print(rmse)

y_pred_flat = y_pred.flatten()

datetime_index = pd.date_range(
    start='2016-12-31 00:10',
    end='2017-01-01 00:00',
    freq='10T')

y_pred_flat = y_pred.flatten()[:len(datetime_index)]

submit_df = pd.DataFrame({
    'datetime': datetime_index,
    'wd (deg)': y_pred_flat
})

submit_df.to_csv(path + 'jena_안태영_submit_Conv1D_1.csv', index=False)

# 11556.9267578125
# 25740.833984375
# 30358.099609375
# 27033.685546875