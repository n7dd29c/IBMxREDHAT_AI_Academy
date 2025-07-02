import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

#1. 데이터
path = ('./Study25/_data/kaggle/bike/bike-sharing-demand/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=7572
)

print(x_train.shape, y_train.shape) # (8708, 8) (8708,)
print(x_test.shape, y_test.shape)   # (2178, 8) (2178,)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = RandomForestRegressor()
# model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# acc :  [0.96666667 0.96666667 0.96666667 0.93333333 0.93333333] 
# 평균 acc :  0.9533

# RandomForestRegressor
# acc :  [0.29171274 0.29440848 0.30464682 0.30724179 0.28058517] 
# 평균 acc :  0.296

# HistGradientBoostingRegressor
# acc :  [0.35473717 0.34454048 0.37182557 0.3544178  0.35079181] 
# 평균 acc :  0.355