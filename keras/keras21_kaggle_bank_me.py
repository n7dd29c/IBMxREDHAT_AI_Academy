import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

le_gen = LabelEncoder()
le_geo = LabelEncoder()

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
test_csv = scaler.transform(test_csv)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
    )
hist = model.fit(x_train, y_train, epochs=500, batch_size=256,
                 validation_split=0.2, callbacks=es, verbose=2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc : ', acc_score)

y_submit = model.predict(test_csv)
submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_bank_me.csv')

#5. 시각화
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('Kaggle Bank')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()

plt.figure(figsize=(9, 6))
plt.plot(hist.history['acc'], c='blue', label='acc_score')
plt.title('Kaggle Bank')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()