import numpy as np
import pandas as pd
import sklearn as sk
import time
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=111
)

scaler = MinMaxScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test))   # -0.06141956477526944 1.0

#2. 모델구성
path = './_save/keras27_mcp/'

# Checkpoint 확인
# model = load_model(path + 'keras27_mcp3.hdf5')

# save_model 확인
model = load_model(path + 'keras27_3_save_model.h5')

#3. 컴파일, 훈련

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
rmse = RMSE(y_test, results)
print('RMSE : ', rmse)

# 11.139924049377441
# 0.9041660960806456
# RMSE :  3.337652445777287