import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
# model = RandomForestRegressor()
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# acc :  [0.96666667 0.96666667 0.96666667 0.93333333 0.93333333] 
# 평균 acc :  0.9533

# RandomForestRegressor
# acc :  [0.80280952 0.80714654 0.79509408 0.81991131 0.8197749 ] 
# 평균 acc :  0.809

# HistGradientBoostingRegressor
# acc :  [0.82665974 0.83756317 0.82428469 0.84326775 0.84739182] 
# 평균 acc :  0.836