#47_0 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score

from xgboost import XGBClassifier

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)

# 1. 데이터

######################################## 결측치 처리 1. 삭제 ########################################
# print(train_csv.isnull().sum())       # 컬럼별 결측치의 갯수 출력
print(train_csv.isna().sum())           # 컬럼별 결측치의 갯수 출력

# train_csv = train_csv.dropna()        # 결측치 제거
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)                      # [1328 rows x 10 columns]

######################################## 결측치 처리 2. 평균값 넣기 ########################################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())

########################################  test_csv 결측치 확인 및 처리 ########################################
# test_csv는 결측치 있을 경우 절대 삭제하면 안된다. 답안지에 해당하는(submission_csv)에 채워넣으려면 갯수가 맞아야한다.
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print('test_csv 정보:', test_csv)
print('ㅡㅡㅡㅡㅡㅡ')

x = train_csv.drop(['count'], axis=1)   # axis = 1 : 컬럼 // axis = 0 : 행
print(x)    # [1459 rows x 9 columns] : count 컬럼을 제거

y = train_csv['count']      # count 컬럼만 추출
print(y.shape)  # (1469,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2,
    # stratify=y
)

# 정규화 전: 원래 컬럼 이름 저장
original_columns = x.columns

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = KNeighborsRegressor(n_neighbors=5)
# model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

print("=================", model.__class__.__name__, "=================" )
# print('acc :', model.score(x_test, y_test))
print('r2 :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score: ', r2)


# y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print('accuracy score: ', acc)
# print('f1 score: ', f1)

'''
================= KNeighborsRegressor =================
r2 : 0.7119187211591961
r2 score:  0.7119187211591961
'''