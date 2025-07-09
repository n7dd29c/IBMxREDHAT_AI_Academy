from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)

train_csv = train_csv.fillna(train_csv.mean())

#################### 결측치 처리 3. 테스트 데이터 ###################
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)   # drop() : 행 또는 열 삭제
                                        # count라는 열(axis=1) 삭제, 참고로 행은 axis=0
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    # stratify=y
)

#2. 모델구성
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # acc :  0.7708296098820614
print(model.feature_importances_)
        
# [0.3736487  0.10615496 0.3357004  0.0250908  0.03824362 0.02943058 0.03338748 0.03666693 0.02167652]
print(np.percentile(model.feature_importances_, 25))
# 0.029430578

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

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
print('acc : ', model.score(x_test, y_test))    # acc :  0.7826047099631074