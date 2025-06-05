import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder, OneHotEncoder

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 완전 비활성화

#1. 데이터
path = './_data/dacon/thyroid/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv) # [87159 rows x 15 columns]
# print(test_csv)  # [46204 rows x 14 columns]

# print(train_csv.isna().sum())   # 결측치 없음
# print(test_csv.isna().sum())   # 결측치 없음

# print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer']


cols = ['Gender', 'Country', 'Race', 'Family_Background', 'Radiation_History',
                        'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes']

train_csv = pd.get_dummies(train_csv, columns=cols)
test_csv = pd.get_dummies(test_csv, columns=cols)

# train/test 열 맞추기
train_csv, test_csv = train_csv.align(test_csv, join='left', axis=1, fill_value=0)

x = train_csv.drop(['Cancer'], axis=1)
# print(x)        # [87159 rows x 14 columns]
y = train_csv['Cancer']
# print(y.shape)  # (87159,)

# test_csv = test_csv.drop(['Race'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv[x.columns])

# print(x_train.dtypes)
# print(y_train.dtype)
# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=34, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
          verbose=1, validation_split=0.1)
end_time = time.time()

#4. 평가, 예측
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있어버리기~')
else:
    print('GPU 없어버리기~')
print('runtime : ', round(end_time - start_time, 2))

# GPU 있어버리기~
# runtime :  705.99