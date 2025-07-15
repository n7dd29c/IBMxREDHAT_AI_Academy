#47_0 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.ensemble import RandomForestClassifier

seed = 123
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
model = RandomForestClassifier(random_state=seed)

model.fit(x_train, y_train)

print("=================", model.__class__.__name__, "=================" )
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy score: ', acc)

# ================= RandomForestClassifier =================
# acc : 0.9649122807017544