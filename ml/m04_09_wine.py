import numpy as np
import pandas as pd

#1. 데이터
from sklearn.datasets import load_wine
datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

print(x_train.shape, y_train.shape) # (142, 13) (142,)
print(x_test.shape, y_test.shape)   # (36, 13) (36,)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
    
# LinearSVC :  1.0
# LogisticRegression :  1.0
# DecisionTreeClassifier :  0.9444444444444444
# RandomForestClassifier :  1.0