import numpy as np
import sklearn as sk
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

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
    test_size=0.1,
    random_state=1998,
)

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=300,
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

# csv 생성
y_submit = model.predict(test_csv)
submission_csv['Outcome'] = np.round(y_submit)
submission_csv.to_csv(path + 'submission_diabetes_test.csv')

# 0527_1253
# loss :  [0.5525025725364685, 0.7175572514533997]
# acc :  0.7175572519083969

# 0527_1259
# loss :  [0.5821629762649536, 0.7251908183097839]
# acc :  0.7251908396946565

# 0527_1309
# loss :  [0.5561426281929016, 0.7295918464660645]
# acc :  0.7295918367346939

# 0527_1315
# loss :  [0.5065860152244568, 0.7938931584358215]
# acc :  0.7938931297709924

# 0527_1317
# loss :  [0.4680461883544922, 0.7938931584358215]
# acc :  0.7938931297709924

# 0527_1433
# loss :  [0.5389078855514526, 0.8015267252922058]
# acc :  0.8015267175572519

# 0527_1446
# loss :  [0.4799043834209442, 0.8167939186096191]
# acc :  0.816793893129771

# 0527_1543
# loss :  0.447693407535553
# acc :  0.803030303030303

# 0527_1552
# loss :  0.42396533489227295
# acc :  0.8636363636363636