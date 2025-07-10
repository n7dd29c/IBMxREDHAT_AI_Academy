#49_06 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor

seed =123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)
feature_names = datasets.feature_names
print(feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'logloss',
    data_name = 'validation_0', 
    # save_best = True,
)

model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = False,
          )

print('acc :', model.score(x_test, y_test))
# print(model.feature_importances_)

# aaa = model.get_booster().get_score(importance_type='weight') # split 빈도수 개념
# {'f0': 5.0, 'f1': 27.0, 'f3': 4.0, 'f4': 10.0, 'f5': 4.0, 'f6': 5.0, 
#  'f7': 20.0, 'f9': 3.0, 'f10': 3.0, 'f12': 2.0, 'f13': 19.0, 'f14': 3.0, 
#  'f15': 12.0, 'f17': 3.0, 'f18': 6.0, ' f19': 3.0, 'f20': 8.0, 'f21': 33.0, 
#  'f22': 10.0, 'f23': 24.0, 'f24': 15.0, 'f25': 1.0, 'f26': 13.0, 
#  'f27': 13.0, 'f28': 9.0, 'f29': 2.0}
aaa = model.get_booster().get_score(importance_type='gain')

# 특성 중요도 값을 배열로 변환
aaa_list = np.array(list(aaa.values()))

# 특성 중요도 값들의 합 계산
aaa_sum = np.sum(aaa_list)

# 정규화된 thresholds 계산 (모든 중요도 값을 합으로 나눔)
thresholds = np.sort(aaa_list / aaa_sum)
print("정규화된 thresholds:", thresholds)
# [0.00167987 0.00189334 0.0019413  0.0036438  0.00377534 0.00448407
#  0.00451154 0.00532142 0.00709919 0.00717433 0.00727892 0.00828942
#  0.00863451 0.00907165 0.00926672 0.01014468 0.01059543 0.0113119
#  0.01505396 0.01716412 0.02565513 0.03090109 0.03786507 0.05862756
#  0.08362495 0.10150925 0.1880387  0.32544274]

# thresholds = np.sort(list(aaa.values())) #오름차순
# print(thresholds)
# [ 0.07786679  0.21503383  0.24158287  0.26567459  0.27211294  0.27391687
#   0.29253069  0.32616228  0.41778874  0.57667625  0.57942659  0.62760967
#   0.64470363  0.66088659  1.00593519  1.10196507  1.16801405  1.79985297
#   1.88523102  2.73181605  2.81559539  3.83561015  6.42857456  7.076159
#  15.39038372 19.13357353]

from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False) 
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다. (기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때. fit 호출하지 않음.
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBClassifier(random_state=seed)
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = False,
          )
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, ACC=%.4f%%'%(i, select_x_train.shape[1], score*100))
    
'''
Trech=0.002, n=28, ACC=97.3684%
Trech=0.002, n=27, ACC=97.3684%
Trech=0.002, n=26, ACC=97.3684%
Trech=0.004, n=25, ACC=97.3684%
Trech=0.004, n=24, ACC=97.3684%
Trech=0.004, n=23, ACC=98.2456%
Trech=0.005, n=22, ACC=97.3684%
Trech=0.005, n=21, ACC=97.3684%
Trech=0.007, n=20, ACC=97.3684%
Trech=0.007, n=19, ACC=96.4912%
Trech=0.007, n=18, ACC=97.3684%
Trech=0.008, n=17, ACC=96.4912%
Trech=0.009, n=16, ACC=97.3684%
Trech=0.009, n=15, ACC=96.4912%
Trech=0.009, n=14, ACC=96.4912%
Trech=0.010, n=13, ACC=97.3684%
Trech=0.011, n=12, ACC=97.3684%
Trech=0.011, n=11, ACC=97.3684%
Trech=0.015, n=10, ACC=97.3684%
Trech=0.017, n=9, ACC=96.4912%
Trech=0.026, n=8, ACC=96.4912%
Trech=0.031, n=7, ACC=96.4912%
Trech=0.038, n=6, ACC=96.4912%
Trech=0.059, n=5, ACC=97.3684%
Trech=0.084, n=4, ACC=98.2456%
Trech=0.102, n=3, ACC=92.9825%
Trech=0.188, n=2, ACC=91.2281%
Trech=0.325, n=1, ACC=90.3509%
'''