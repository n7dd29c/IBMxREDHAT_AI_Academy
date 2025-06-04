import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
#  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.1, random_state=337
# )
# 이렇게 쓰면 y에 라벨값이 불균형하게 들어갈 수 있다 (line 26)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#2. 모델구성
modelpath = './_save/keras28_mcp/09_wine/'
model = load_model(modelpath + 'k28_250604_1256_0200-0.0612.hdf5')

#3. 컴파일, 훈련

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)

# loss :  0.09748025238513947
# acc :  0.9166666865348816
# f1_score :  0.9181671790367444

# loss :  0.09748025238513947
# acc :  0.9166666865348816
# f1_score :  0.9181671790367444