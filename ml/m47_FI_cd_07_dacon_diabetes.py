from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    stratify=y
)

#2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # acc :  0.7022900763358778
print(model.feature_importances_)
        
# [0.09888884 0.28201106 0.08759275 0.07843267 0.08501508 0.14154954 0.09348151 0.1330285 ]
print(np.percentile(model.feature_importances_, 25))
# 0.086948335

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

col_name = []
# ['SkinThickness', 'Insulin']

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
    stratify=y
)

model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test))    # acc :  0.6946564885496184