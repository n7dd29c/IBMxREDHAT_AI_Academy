import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']
print(x)
print(y)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True)

# 이렇게도 되고 (쉬운방법)
# for train_index, val_index in kfold.split(df):
#     print('=========================================================================')
#     print(train_index, '\n', val_index)

# 이렇게도 된다 (복잡한방법)
for num, (train_index, val_index) in enumerate(kfold.split(df)):
    print('===================================',num,'===================================')
    print(train_index, '\n', val_index)

