import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

#1. 데이터
path = './Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [715 rows x 9 columns]

train_csv = train_csv.fillna(train_csv.mean())


#################### 결측치 처리 3. 테스트 데이터 ###################
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)   # drop() : 행 또는 열 삭제
                                        # count라는 열(axis=1) 삭제, 참고로 행은 axis=0
# print(x)                                # [1459 rows x 9 columns]


y = train_csv['count']                  # count 컬럼만 빼서 y에 대입
# print(y.shape)                          # (1459,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=444
)

print(x_train.shape, y_train.shape) # (1167, 9) (1167,)
print(x_test.shape, y_test.shape)   # (292, 9) (292,)

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
# acc :  [0.76514664 0.79435775 0.83295653 0.72713644 0.80099346] 
# 평균 acc :  0.784

# HistGradientBoostingRegressor
# acc :  [0.78164692 0.80369098 0.8278023  0.72471007 0.81240662] 
# 평균 acc :  0.79