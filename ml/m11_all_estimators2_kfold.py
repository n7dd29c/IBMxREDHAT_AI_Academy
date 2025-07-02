# bank

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)

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
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, shuffle=True
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True)

#2. 모델구성
allAlgorithms= all_estimators(type_filter='regressor',)
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))

max_score = 0
max_name = 'name'

for (name, algorithm) in allAlgorithms:
    try: 
        model = algorithm()
        
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print('score : ', np.round(np.mean(score), 3))
        
        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_pred)
        print('ACC : ', acc)
        
        if max_score<acc:
            max_score = acc
            max_name = name
        
    except:
        print(name,'은(는) 에러')
        
        
print('===============================================')
print('최고성능모델 : ', max_name, max_score)
print('===============================================')

# ===============================================
# 최고성능모델 :  HistGradientBoostingRegressor 0.7882931268302555
# ===============================================