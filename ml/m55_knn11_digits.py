#47_0 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score

from xgboost import XGBClassifier

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2,
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
f1 = f1_score(y_test, y_pred, average='macro')
print('accuracy score: ', acc)
print('f1 score: ', f1)

'''
================= KNeighborsClassifier =================
acc : 0.9833333333333333
accuracy score:  0.9833333333333333
f1 score:  0.9832414016496175
'''