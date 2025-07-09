from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed=55
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = ('./Study25/_data/kaggle/bike/bike-sharing-demand/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

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

model = XGBRegressor(random_state=seed,)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],)
print('=============', model.__class__.__name__, '=============')
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
    print(select_x_train.shape)
    
    select_model = XGBRegressor(random_state=seed,)
    
    select_model.fit(select_x_train, y_train,
                    eval_set = [(select_x_test, y_test)], verbose=1)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, R2=%.4f%%'%(i, select_x_train.shape[1], score*100))