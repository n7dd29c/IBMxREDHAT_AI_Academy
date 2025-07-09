from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    stratify=y
)

#2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # acc :  1.0
print(model.feature_importances_)
        
# [0.03466775 0.04431445 0.00974758 0.03754259 0.01700081 0.01926046
#  0.12481797 0.00133649 0.01961179 0.1467493  0.01939317 0.39759117
#  0.12796643]
print(np.percentile(model.feature_importances_, 25))
# 0.019260464

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

col_name = []
# ['ash', 'magnesium', 'total_phenols', 'nonflavanoid_phenols']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x_f = x_f.drop(columns=col_name)
print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_f, y, test_size=0.2, random_state=seed,
    stratify=y
)

model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test))    # acc :  1.0