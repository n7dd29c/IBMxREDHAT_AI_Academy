import time
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)
print(y)
# print(np.unique(y, return_counts=True))
                            # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.DataFrame(y).value_counts())
                            # 0
                            # 0    50
                            # 1    50
                            # 2    50
                            # Name: count, dtype: int64
# print(pd.value_counts(y))
                            # 0
                            # 0    50
                            # 1    50
                            # 2    50
                            # Name: count, dtype: int64

############# OneHotEncoding #############


#1. sklearn
from sklearn.preprocessing import OneHotEncoder
''' 희소행렬방식
# y = y.reshape(-1, 1)    # 벡터형태의 데이터를 매트릭스 형태로 변환
# ohe = OneHotEncoder()   # 매트릭스 형태로 받기때문에 N,1로 reshape하고 해야한다
# y_encoded1 = ohe.fit_transform(y_encoded1)
# print(y_encoded1)
# print(y_encoded1.shape)          # (150, 3)
# print(type(y_encoded1))          # <class 'scipy.sparse.csr.csr_matrix'>
# y_encoded1 = y_encoded1.toarray()         # scipy를 numpy를 변환
# print(type(y_encoded1))          # <class 'numpy.ndarray'>
'''
# y = y.reshape(-1, 1)    # 벡터형태의 데이터를 매트릭스 형태로 변환
# ohe = OneHotEncoder(sparse=False)  # numpy형태로 출력, defalut = False
# y = ohe.fit_transform(y)
print(y.shape)

# region pandas, keras

# #2. pandas
y = pd.get_dummies(y)  # DataFrame → numpy 배열
# print(y_encoded2)
# print(y_encoded2.shape) # (150, 3)

# #3. keras, 0부터 시작하지 않으면 빈 0 컬럼을 생성해버림
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) # (150, 3)

# endregion

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=111
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=50, restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.2, callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])
print(results[1])
y_pred = model.predict(x_test)
print(end_time - start_time)

# 0.10592655092477798
# 0.9333333373069763

######### accuracy score를 사용해서 출력 #########
from sklearn.metrics import accuracy_score

# onehotencoder
# y_pred = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)
# acc_score = accuracy_score(y_true, y_pred)
# print(acc_score)

# getdummy
y_pred = np.argmax(y_pred, axis=1)
y_test = y_test.values
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))