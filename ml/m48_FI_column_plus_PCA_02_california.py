from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)

#2. 모델구성
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('R2 origin : ', model.score(x_test, y_test))    # R2 :  0.842170152144573
print(model.feature_importances_)
        
# [0.47785544 0.06601944 0.04326427 0.02447754 0.02543438 0.1539704 0.09944526 0.10953324]
print(np.percentile(model.feature_importances_, 25))
# 0.0388068

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

col_name = []
# ['AveBedrms', 'Population']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)
x2 = x_f[['AveBedrms', 'Population']]

x1_train, x1_test = train_test_split(
    x1, test_size=0.2, random_state=seed,
)

model.fit(x1_train, y_train)
print('R2 drop : ', model.score(x1_test, y_test))    # R2 drop :  0.843088054071279

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size=0.2, random_state=seed,
)
print(x1_train.shape, x1_test.shape)    # (16512, 6) (4128, 6)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape)    # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)    # (16512, 7) (4128, 7)

model.fit(x_train, y_train)
print('R2 PCA : ', model.score(x_test, y_test))    # R2 PCA :  0.841787590830198