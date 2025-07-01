import numpy as np
import pandas as pd

#1. 데이터
path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

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

# LinearSVC :  0.7786259541984732
# LogisticRegression :  0.7786259541984732
# DecisionTreeClassifier :  0.7480916030534351
# RandomForestClassifier :  0.7862595419847328