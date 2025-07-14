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
               n_init=10, random_state=seed
               )

from sklearn.metrics import confusion_matrix

# 클러스터 예측
y_train_pred = model.fit_predict(x_train)

# confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", cm)

# 정확하게 대응되는 방향 선택
acc1 = accuracy_score(y_train, y_train_pred)
acc2 = accuracy_score(y_train, 1 - y_train_pred)

if acc1 > acc2:
    print("클러스터 라벨은 실제 y와 동일하게 매핑됨")
    y_aligned = y_train_pred
else:
    print("클러스터 라벨은 실제 y와 반대로 매핑됨 (0↔1)")
    y_aligned = 1 - y_train_pred

# 정렬된 예측값으로 최종 정확도
print("최종 Accuracy:", accuracy_score(y_train, y_aligned))