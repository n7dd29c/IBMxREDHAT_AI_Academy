import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
path = './Study25/_data/dacon/thyroid/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

cols = ['Gender', 'Country', 'Race', 'Family_Background', 'Radiation_History',
                        'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes']

train_csv = pd.get_dummies(train_csv, columns=cols)
test_csv = pd.get_dummies(test_csv, columns=cols)

# train/test 열 맞추기
train_csv, test_csv = train_csv.align(test_csv, join='left', axis=1, fill_value=0)

x = train_csv.drop(['Cancer'], axis=1)
# print(x)        # [87159 rows x 14 columns]
y = train_csv['Cancer']
# print(y.shape)  # (87159,)

# test_csv = test_csv.drop(['Race'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv[x.columns])

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = RandomForestClassifier()
# model = HistGradientBoostingClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))

# acc :  [0.96666667 0.96666667 0.96666667 0.93333333 0.93333333] 
# 평균 acc :  0.9533

# RandomForestClassifier
# acc :  [0.88194126 0.88050711 0.8815397  0.88142497 0.8830245 ] 
# 평균 acc :  0.882

# HistGradientBoostingClassifier
# acc :  [0.88142497 0.88004819 0.88303121 0.88090867 0.88130342] 
# 평균 acc :  0.881