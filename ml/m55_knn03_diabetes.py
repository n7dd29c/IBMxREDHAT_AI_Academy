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

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)

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
r2 : 0.45373900245244714
r2 score:  0.45373900245244714
'''