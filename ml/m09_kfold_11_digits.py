import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

#1. 데이터
x, y = load_digits(return_X_y=True)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
# model = RandomForestRegressor()
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# acc :  [0.96666667 0.96666667 0.96666667 0.93333333 0.93333333] 
# 평균 acc :  0.9533

# RandomForestRegressor
# acc :  [0.34363882 0.43203779 0.32370365 0.45741769 0.57380742] 
# 평균 acc :  0.426

# HistGradientBoostingRegressor
# acc :  [0.2406803  0.33171027 0.33067035 0.4442867  0.5908704 ] 
# 평균 acc :  0.388