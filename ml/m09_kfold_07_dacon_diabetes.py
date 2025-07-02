import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

#1. 데이터
path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

y = train_csv['Outcome']
print(x)        # [652 rows x 8 columns]
print(y.shape)  # (652,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1998,
)

print(x_train.shape, y_train.shape) # (521, 8) (521,)
print(x_test.shape, y_test.shape)   # (131, 8) (131,)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = RandomForestClassifier()
# model = HistGradientBoostingClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# RandomForestClassifier
# acc :  [0.78625954 0.82442748 0.75384615 0.7        0.68461538] 
# 평균 acc :  0.75

# HistGradientBoostingClassifier
# acc :  [0.71755725 0.78625954 0.70769231 0.69230769 0.73846154] 
# 평균 acc :  0.728