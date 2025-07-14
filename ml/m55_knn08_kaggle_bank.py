#47_0 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score

from xgboost import XGBClassifier

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1.데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 문자 데이터 수치화(인코딩)
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# # 아래 2줄이랑 같다.
# le_geo.fit(train_csv['Geography'])                                    # 'Geography' 컬럼을 기준으로 인코딩한다.
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])     # 적용하고 train_csv['컬럼']에 입력함.
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])

# 테스트 데이터도 수치화해야한다. 위에서 인스턴스가 이미 fit해놨기때문에 transform만 적용한다.
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)  
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = KNeighborsRegressor(n_neighbors=5)
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

print("=================", model.__class__.__name__, "=================" )
print('acc :', model.score(x_test, y_test))
# print('r2 :', model.score(x_test, y_test))

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print('r2 score: ', r2)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy score: ', acc)
print('f1 score: ', f1)

'''
================= KNeighborsClassifier =================
acc : 0.8470021510588663
accuracy score:  0.8470021510588663
f1 score:  0.5952228278294325
'''