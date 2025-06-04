import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

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

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv[x.columns])

# print(x_train.dtypes)
# print(y_train.dtype)
# exit()

#2. 모델구성
modelpath = './_save/keras28_mcp/06_cancer/'
model = load_model(modelpath + 'k28_250604_1231_0004-0.3076.hdf5')

#3. 컴파일, 훈련

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss : ', results[0])
y_predict = np.round(y_predict)
f1 = f1_score(y_test, y_predict)
print('F1_score', f1)

# loss :  0.30811169743537903
# F1_score :  0.46113719353155974

# loss :  0.30811169743537903
# F1_score 0.46113719353155974