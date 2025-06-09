


from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x)    # (442, 10)
# print(y)    # (442,)
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=345
)

scaler = StandardScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(16, input_dim = 10, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

input = Input(shape=(10,))
dense1 = Dense(16, activation='relu')(input)
dense2 = Dense(32, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense5)
dense6 = Dense(32, activation='relu')(drop4)
drop5 = Dropout(0.2)(dense6)
dense7 = Dense(32, activation='relu')(dense1)
drop6 = Dropout(0.2)(dense7)
dense8 = Dense(32, activation='relu')(drop6)
drop7 = Dropout(0.2)(dense8)
dense9 = Dense(32, activation='relu')(drop7)
dense10 = Dense(16, activation='relu')(dense9)
output = Dense(1)(dense10)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

r2 = r2_score(y_test, results)
print(r2)

# StandardScaler
# 0.5113784689346782

# dropout
# 0.40730772991171227