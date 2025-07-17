from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# pf = PolynomialFeatures(degree=2, include_bias=False)
# x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)

#2. 모델구성
es = xgb.callback.EarlyStopping(
    rounds = 20,
    metric_name = 'mae',
    # data_name = 'validation_0',
    # save_best = True
)

model = XGBRegressor(
    random_state=seed,
    n_estimators = 10000,
    # max_depth = 0,
    # gamma = 0,
    # min_child_weight = 0,
    # subsample = 0.4,
    # reg_alpha = 0,
    # reg_lambda = 0,
    eval_metric = 'mae',   # 다중분류 : mlogloss, merror / 이진분류 : logloss, error / 회귀 : rmse, mae, rmsle
    #                             # 2.1.1 버전 이후로 fit에서 모델로 위치이동
    callbacks = [es]
    )

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],)
print('=============', model.__class__.__name__, '=============')
print('acc : ', model.score(x_test, y_test))

# PF 적용 전
# acc :  0.8880192041397095

# PF 적용 후
# acc :  0.8938437700271606