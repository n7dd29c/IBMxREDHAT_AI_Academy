import sklearn as sk
import tensorflow as tf
import numpy as np

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 인증서 오류 처리

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,               # 각각 train과 test에 들어갈 값
    test_size=0.1,      # 전체 데이터 중 테스트데이터에 쓸 비율, 학습데이터는 자동으로 나머지로 정해짐
    random_state=748    # 랜덤시드값
)

#2. 모델구성

path = './_save/keras28_mcp/02_california/'

model = load_model(path + 'k28_250604_1115_0064-0.4917.hdf5')

#3. 컴파일, 훈련

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)

# 0.6561379082601324
# 0.6561379082601324