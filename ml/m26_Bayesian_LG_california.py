from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import lightgbm as lgb
from lightgbm import early_stopping
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random

seed = 55
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
parameters = {
   'n_estimators':(100,500),
   'learning_rate':(0.0001, 0.5),
   'max_depth':(3,10),
   'num_leaves':(24,40),
   'min_child_samples':(10,200),
   'min_split_gain':(0,5),
   'subsample':(0.5,1.0),
   'colsample_bytree':(0.5,1),
   'max_bin':(9,500),
   'reg_lambda':(0,100),    # default : 1 // L2정규화 // 릿지
   'reg_alpha':(0,10)       # default : 0 // L1정규화 // 랏쏘
}

def lgb_function(n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, min_split_gain,
                  subsample, colsample_bytree, reg_lambda, reg_alpha, max_bin):
    params = {'n_estimators':int(round(n_estimators)),
              'learning_rate':learning_rate,
              'max_depth':int(round(max_depth)),
              'num_leaves':int(round(num_leaves)),
              'min_child_samples':int(round(min_child_samples)),
              'min_split_gain':min_split_gain,
              'subsample':subsample,
              'colsample_bytree':colsample_bytree,
              'reg_lambda':reg_lambda,
              'reg_alpha':reg_alpha,
              'max_bin':int(round(max_bin)),
              }
    
    callbacks = [early_stopping(stopping_rounds=20, verbose=False)]
    model = lgb.LGBMRegressor(**params, n_jobs=-1, verbosity=-1)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], callbacks=callbacks, )

    y_pred = model.predict(x_test)
    results = r2_score(y_test, y_pred)
    
    return results

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = lgb_function,
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