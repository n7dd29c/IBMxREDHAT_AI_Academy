from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# print(x.shape)  # (581012, 54)
# print(y.shape)  # (581012,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, #stratify=y
)

print(x_train.shape, y_train.shape) # (464809, 54) (464809,)
print(x_test.shape, y_test.shape)   # (116203, 54) (116203,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (464809, 7) (116203, 7)

x_train = x_train.reshape(-1, 18, 3)
x_test = x_test.reshape(-1, 18, 3)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(18,3)))
model.add(MaxPooling1D())
model.add(Dropout(0.3))
model.add(Conv1D(32, 2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs = 100, batch_size = 2048,
          verbose=2, validation_split=0.2, callbacks=es)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_round = np.round(y_predict)
f1 = f1_score(y_test, y_round, average='macro')
print('f1 : ', f1)
print(end_time-start_time)

# loss :  0.3646397292613983
# acc :  0.8459936380386353
# f1 :  0.7007749003381016

# loss :  0.38191795349121094
# acc :  0.8386960625648499
# f1 :  0.7130039941371872
# 217.774178981781

# loss :  0.4580861032009125
# acc :  0.8067433834075928
# f1 :  0.6695642879170728
# 1873.2197699546814