import numpy as np
import sklearn as sk
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  StandardScaler

#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)   # [116 rows x 1 columns]

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

y = train_csv['Outcome']
print(x)        # [652 rows x 8 columns]
print(y.shape)  # (652,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1998,
)

print(x_train.shape, y_train.shape) # (521, 8) (521,)
print(x_test.shape, y_test.shape)   # (131, 8) (131,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 2, 2, 2)
x_test = x_test.reshape(-1, 2, 2, 2)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, 2, 1, input_shape=(2,2,2), padding='same'))
model.add(Conv2D(32, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=10000, batch_size=128,
          validation_split=0.2, callbacks=[es], verbose=3)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('acc : ', acc_score)

# StandardScaler
# loss :  0.7154603600502014
# acc :  0.7424242424242424

# dropout
# loss :  0.4110744595527649
# acc :  0.7727272727272727

# loss :  0.3985040485858917
# acc :  0.8167939186096191