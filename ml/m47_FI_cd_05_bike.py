from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = ('./Study25/_data/kaggle/bike/bike-sharing-demand/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    # stratify=y
)

#2. 모델구성
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # 0.3032136559486389
print(model.feature_importances_)
        
# [0.12758577 0.05774558 0.09566985 0.06467694 0.10418771 0.34981766 0.14385027 0.05646621]
print(np.percentile(model.feature_importances_, 25))
# 0.0629441

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float64'>

col_name = []
# ['sepal width (cm)']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
print(col_name)

x_f = x.drop(columns=col_name)
print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_f, y, test_size=0.2, random_state=seed,
    # stratify=y
)

model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test))    # acc :  0.30135852098464966