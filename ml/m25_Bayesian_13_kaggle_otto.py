from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed=123

#1. 데이터
path = './Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# print(train_csv.shape)  # (200000, 201)
# print(test_csv.shape)   # (200000, 200)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
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
    
    model = XGBClassifier(**params, n_jops=-1, eval_metric='merror', early_stopping_rounds=20,)
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