# california

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, shuffle=True
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True)

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
        r2 = r2_score(y_test, y_pred)
        print('R2 : ', r2)
        
        if max_score<r2:
            max_score = r2
            max_name = name
        
    except:
        print(name,'은(는) 에러')
        
        
print('===============================================')
print('최고성능모델 : ', max_name, max_score)
print('===============================================')

# ===============================================
# 최고성능모델 :  HistGradientBoostingRegressor 0.7882931268302555
# ===============================================