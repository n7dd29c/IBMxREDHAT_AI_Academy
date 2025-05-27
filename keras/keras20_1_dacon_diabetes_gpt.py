import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan).fillna(x.mean())
test_csv = test_csv.replace(0, np.nan).fillna(test_csv.mean())
y = train_csv['Outcome']

# 🔹 정규화
scaler = StandardScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=200,
    restore_best_weights=True
)

model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=3
)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"loss : {loss:.4f}")
print(f"acc  : {acc:.4f}")

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc_score)

# 5. 제출 파일 생성
y_submit = model.predict(test_csv)
submission_csv['Outcome'] = np.round(y_submit)
submission_csv.to_csv(path + 'submission_diabetes_final.csv')
