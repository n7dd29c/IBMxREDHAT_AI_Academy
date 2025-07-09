from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)

#2. 모델구성
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # acc :  0.38520249164165343
print(model.feature_importances_)
        
print(np.percentile(model.feature_importances_, 25))
# 0.0015794792

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

col_name = []
# ['age', 'sex', 's1']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x_f = x_f.drop(columns=col_name)
print(x)

x_train, x_test = train_test_split(
    x_f, test_size=0.2, random_state=seed,
)

model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test))    # acc :  0.31954334686275887