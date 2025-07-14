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
path = './_data/kaggle/bike/'           # 상대경로 : 대소문자 구분X

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

####### x와 y 분리 ######
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)    # test셋에는 없는 casual, registered, y가 될 count는 x에서 제거
y = train_csv['count']

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
r2 : 0.2532415215879523
r2 score:  0.2532415215879523
'''