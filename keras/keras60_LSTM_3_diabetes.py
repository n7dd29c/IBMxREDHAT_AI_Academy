from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x)    # (442, 10)
# print(y)    # (442,)
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=345
)

print(x_train.shape, y_train.shape) # (353, 10) (353,)
print(x_test.shape, y_test.shape)   # (89, 10) (89,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(10,1), return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

r2 = r2_score(y_test, results)
print(r2)

# 0.5822236648357441

# 0.3145710271715726