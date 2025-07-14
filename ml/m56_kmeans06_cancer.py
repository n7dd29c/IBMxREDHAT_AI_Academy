#KMEans는 무조건 분류형

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
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

seed = 56
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = KMeans(n_clusters=2, init='k-means++', 
               n_init=15, random_state=seed
               )

y_train_pred = model.fit_predict(x_train)
y_train_reshape = y_train.reshape(-1,1)
y_train_pred = model.fit_predict(y_train_reshape)

print(y_train_pred[:10])
print(y_train[:10])

exit()

print("=================", model.__class__.__name__, "=================" )
print('acc :', model.score(x_test, y_test))
# print('r2 :', model.score(x_test, y_test))

'''
================= KNeighborsClassifier =================
acc : 0.9833333333333333
accuracy score:  0.9833333333333333
f1 score:  0.9832414016496175
'''