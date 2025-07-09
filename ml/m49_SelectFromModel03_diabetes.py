from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed=123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    # stratify=y
)

#2. 모델구성
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'mae',
    data_name = 'validation_0',
    # save_best = True
)

model = XGBRegressor(
    random_state=seed,
    # n_estimators = 500,
    # max_depth = 0,
    # gamma = 0,
    # min_child_weight = 0,
    # subsample = 0.4,
    # reg_alpha = 0,
    # reg_lambda = 0,
    # eval_metric = 'mae',   # 다중분류 : mlogloss, merror / 이진분류 : logloss, error / 회귀 : rmse, mae, rmsle
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동
    # callbacks = [es]
    )

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))    # acc :  0.9333333333333333
print(model.feature_importances_)
        
# [0.01920634 0.01366225 0.83778393 0.12934752]
print(np.percentile(model.feature_importances_, 25))
# 0.01782032

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float64'>

col_name = []
# ['sepal width (cm)']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x_f = x_f.drop(columns=col_name)
print(x_f)

x_train, x_test, y_train, y_test= train_test_split(
    x_f, y, test_size=0.2, random_state=seed,
    # stratify=y
)

model.fit(x_train, y_train, eval_set = [(x_test, y_test)],)
print('acc : ', model.score(x_test, y_test))    # acc :  0.9333333333333333

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_) # 오름차순이 default
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    select = SelectFromModel(model, threshold=i, prefit=False)
    # threshold가 i값 이상인것을 모두 훈련시킨다
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련 (기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때
    
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)
    # print(select_x_train.shape)
    
    select_model = XGBRegressor(
        random_state=seed,
        # n_estimators = 500,
        # max_depth = 0,
        # gamma = 0,
        # min_child_weight = 0,
        # subsample = 0.4,
        # reg_alpha = 0,
        # reg_lambda = 0,
        # eval_metric = 'mae',    # 다중분류 : mlogloss, merror / 이진분류 : logloss, error
                                    # 2.1.1 버전 이후로 fit에서 모델로 위치이동
        # callbacks = [es]
    )
    
    select_model.fit(select_x_train, y_train,
                    eval_set = [(select_x_test, y_test)], verbose=False)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, R2=%.4f%%'%(i, select_x_train.shape[1], score*100))