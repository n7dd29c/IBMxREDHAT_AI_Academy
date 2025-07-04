from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
import time
import pandas as pd
import numpy as np
import random   
import warnings
warnings.filterwarnings('ignore')

seed = 55
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)

train_csv = train_csv.fillna(train_csv.mean())


#################### 결측치 처리 3. 테스트 데이터 ###################
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)   # drop() : 행 또는 열 삭제
                                        # count라는 열(axis=1) 삭제, 참고로 행은 axis=0
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

parameters = {
   'n_estimators':(100,500),
   'learning_rate':(0.0001, 0.5),
   'max_depth':(3,10),
#    'num_leaves':(24,40),
#    'min_child_sample':(10,200),
   'min_child_weight':(1,10),
   'gamma':(0,5),
   'subsample':(0.5,2),
   'colsample_bytree':(0.5,1),
   'colsample_bylevel':(0.5,1),
#    'max_bin':(9,500),
   'reg_lambda':(0,100),    # default : 1 // L2정규화 // 릿지
   'reg_alpha':(0,10)       # default : 0 // L1정규화 // 랏쏘
}

def xgb_function(n_estimators, learning_rate, max_depth, min_child_weight, gamma,
                 subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha):
    params = {'n_estimators':round(n_estimators),
              'learning_rate':learning_rate,
              'max_depth':int(round(max_depth)),
              'min_child_weight':int(round(min_child_weight)),
              'gamma':gamma,
              'subsample':max(min(subsample,1),0),  # 0.5에서 2사이를 0과 1사이로 변환
              'colsample_bytree':colsample_bytree,
              'colsample_bylevel':colsample_bylevel,
              'reg_lambda':max(reg_lambda, 0),
              'reg_alpha':reg_alpha,
              }
    
    model = XGBRegressor(**params, n_jops=-1, eval_metric='rmse', early_stopping_rounds=20,)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)

    y_pred = model.predict(x_test)
    results = r2_score(y_test, y_pred)
    
    return results

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = xgb_function,
    pbounds=parameters,
    random_state=seed,
    verbose=2
)

n_iter = 100
start = time.time()
optimizer.maximize(init_points=5, n_iter=n_iter)
end = time.time() - start

print(optimizer.max)
print(n_iter, '번 걸린 시간 : ', round(end), '초')