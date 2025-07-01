import numpy as np
import pandas as pd

#1. 데이터
path = './Study25/_data/dacon/thyroid/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

cols = ['Gender', 'Country', 'Race', 'Family_Background', 'Radiation_History',
                        'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes']

train_csv = pd.get_dummies(train_csv, columns=cols)
test_csv = pd.get_dummies(test_csv, columns=cols)

x = train_csv.drop(['Cancer'], axis=1)
# print(x)        # [87159 rows x 14 columns]
y = train_csv['Cancer']
# print(y.shape)  # (87159,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

print(x_train.shape, y_train.shape) # (69727, 34) (69727,)
print(x_test.shape, y_test.shape)   # (17432, 34) (17432,)

''' 주의!!! '''
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
''' 원래 쓰면 안됨 '''

#2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## 최종결과예시 -> ##
# LinearSCV : 0.7
# LogisticRegression : 0.8
# DecisionTreeClassifier : 0.9
# RandomForestClassifier : 1.0

model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]
model_name = ['LinearSVC', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']

for i,model_list in enumerate(model_list):
    model = model_list()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f'{model_name[i]} : ', score)
    
# LinearSVC :  0.8799908214777421
# LogisticRegression :  0.880277650298302
# DecisionTreeClassifier :  0.8145938503900872
# RandomForestClassifier :  0.8810807709958697