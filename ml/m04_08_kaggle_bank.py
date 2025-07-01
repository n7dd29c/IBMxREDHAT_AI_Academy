import numpy as np
import pandas as pd

#1. 데이터
path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 문자 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv.shape)  # (165034, 11)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=111,
)

print(x_train.shape, y_train.shape) # (132027, 10) (132027,)
print(x_test.shape, y_test.shape)   # (33007, 10) (33007,)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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
    
# LinearSVC :  0.8246129608870846
# LogisticRegression :  0.8284606295634259
# DecisionTreeClassifier :  0.7978610597751992
# RandomForestClassifier :  0.860544732935438