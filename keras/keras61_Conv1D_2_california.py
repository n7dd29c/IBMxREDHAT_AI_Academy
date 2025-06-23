from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,               # 각각 train과 test에 들어갈 값
    test_size=0.1,      # 전체 데이터 중 테스트데이터에 쓸 비율, 학습데이터는 자동으로 나머지로 정해짐
    random_state=748    # 랜덤시드값
)

print(x_train.shape, y_train.shape) # (18576, 8) (18576,)
print(x_test.shape, y_test.shape)   # (2064, 8) (2064,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(8,1)))
model.add(MaxPooling1D())
model.add(Conv1D(32, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=2
)
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)

# 0.7869105920229292
# runtime :  36.35

# 0.7584652257919049