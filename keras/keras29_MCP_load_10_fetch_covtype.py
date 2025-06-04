import ssl
import certifi

# SSL 인증서 문제 해결
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
# print(np.unique(y, return_counts=True))
# print(pd.value_counts(y, sort=True))    # True는 빈도순으로 정렬, False는 데이터 순서

# print(datasets.feature_names)
# ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
#  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
#  'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
#  'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3',
#  'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5',
#  'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11',
#  'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16',
#  'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21',
#  'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26',
#  'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31',
#  'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36',
#  'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']

# print(type(x))  # <class 'numpy.ndarray'>
# print(type(y))  # <class 'numpy.ndarray'>

# print(x.shape)  # (581012, 54)
# print(y.shape)  # (581012,)

y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, #stratify=y
)

#2. 모델구성
modelpath = './_save/keras28_mcp/10_fetch_covtype/'
model = load_model(modelpath + 'k28_250604_1302_0120-0.1579.hdf5')

#3. 컴파일, 훈련

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_round = np.round(y_predict)
f1 = f1_score(y_test, y_round, average='macro')
print('f1 : ', f1)

# loss :  0.15752246975898743
# acc :  0.9387451410293579
# f1 :  0.9012750500310107

# loss :  0.15752246975898743
# acc :  0.9387451410293579
# f1 :  0.9012750500310107