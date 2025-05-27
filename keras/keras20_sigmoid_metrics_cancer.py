import numpy as np
import pandas as pd
import sklearn as sk
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)           # 데이터 명세
print(datasets.feature_names)   # 컬럼 이름
print(type(datasets))           # <class 'sklearn.utils._bunch.Bunch'>

x = datasets.data               # dictionary = key value
y = datasets.target

print(x.shape)  # (569, 30)
print(y.shape)  # (569,)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>
print(x)
print(y)

# 0과 1의 개수가 몇개인지 찾아보기

# numpy의 경우
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

# pandas의 경우
print(pd.value_counts(y))   # 1    357
                            # 0    212
                            
print(pd.DataFrame(y).value_counts())   # 행렬(매트릭스)형태의 데이터는 DataFrame
print(pd.Series(y).value_counts())      # 벡터형태의 데이터는 Series

# 지금처럼 데이터의 불균형(357, 212)이 있을땐 데이터 증폭기법의 사용을 고려해야한다

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=7275,
    shuffle=True,
)

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',         # 모니터 할 값
    mode='min',                 # 최대값 max, 알아서 찾기 auto
    patience=80,                # patience 안에 최소값이 안나오면 멈춤
    restore_best_weights=True,  # 최소값을 저장할거면 True, default는 False
    )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 32,
                 verbose=1, validation_split=0.2, callbacks=[es],)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
# print(results)  # [0.07660776376724243, 0.9590643048286438]
#                 # binary_crossentropy값 | accuracy 값
print('loss : ', round(results[0], 4))      # loss : 0.0766
print('accuracy : ', round(results[1], 4))  # accuracy : 0.9766

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', round(acc_score, 4))

print('runtime : ', round(end_time - start_time, 2), 'sec')


plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.plot(hist.history['acc'], c='yellow', label='acc')
plt.title('Kaggle Bank')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()