from sklearn.datasets import load_iris
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
datasets = load_iris()
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
print('acc : ', model.score(x_test, y_test))    # acc :  0.9333333333333333
print(model.feature_importances_)
        
# [0.01920634 0.01366225 0.83778393 0.12934752]
print(np.percentile(model.feature_importances_, 25))
# 0.01782032

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float64'>

col_name = []
# ['sepal width (cm)']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x_f = x_f.drop(columns=col_name)
print(x_f)

x_train, x_test, y_train, y_test= train_test_split(
    x_f, y, test_size=0.2, random_state=seed,
    stratify=y
)

model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test))    # acc :  0.9333333333333333