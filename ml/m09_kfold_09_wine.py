import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

#1. 데이터
x, y = load_wine(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = RandomForestRegressor()
# model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# acc :  [0.96666667 0.96666667 0.96666667 0.93333333 0.93333333] 
# 평균 acc :  0.9533

# RandomForestRegressor
# acc :  [0.87962761 0.91971255 0.92391663 0.96344234 0.91554595] 
# 평균 acc :  0.92

# HistGradientBoostingRegressor
# acc :  [0.85756686 0.89868648 0.90547983 0.94301781 0.94212047] 
# 평균 acc :  0.909