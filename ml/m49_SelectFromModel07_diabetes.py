from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    stratify=y
)

#2. 모델구성
es = xgb.callback.EarlyStopping(
    rounds = 50,
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True
)

model = XGBClassifier(random_state=seed,)

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
    # print(select_x_train.shape)
    
    select_model = XGBClassifier(random_state=seed,)
    
    select_model.fit(select_x_train, y_train,
                    eval_set = [(select_x_test, y_test)], verbose=False)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, ACC=%.4f%%'%(i, select_x_train.shape[1], score*100))