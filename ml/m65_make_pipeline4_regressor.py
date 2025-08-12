# 실습
# california, diabetes, ddarung, bike / 4개의 데이터셋을
# pipeline을 통해 여러 모델과 여러 스케일러를 비교하고
# 제일 성능이 좋았을 때 어떤 모델과 스케일러를 썼는지, 점수 출력

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor as xgb
from lightgbm import LGBMRegressor as lgb
from catboost import CatBoostRegressor as cat

import warnings
warnings.filterwarnings('ignore')

#1. 데이터

############################################################ california
datasets = fetch_california_housing()
x1 = datasets.data
y1 = datasets.target

############################################################ diabetes
x2, y2 = load_diabetes(return_X_y=True)
print(x2.shape, y2.shape) # (569, 30) (569,)

############################################################ ddarung
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x3 = train_csv.drop(['count'], axis=1)
y3 = train_csv['count']

############################################################ bike
path = ('./_data/kaggle/bike/bike-sharing-demand/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x4 = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y4 = train_csv['count']

datasets = {
    'california':   (x1, y1),
    'diabetes':     (x2, y2),
    'ddarung':      (x3, y3),
    'bike':         (x4, y4),
}

scalers = {
    'MinMax':    MinMaxScaler(),
    'Standard':  StandardScaler(),
    'Robust':    RobustScaler(),
    'MaxAbs':    MaxAbsScaler(),
}

models = {
    'XGBRegressor': xgb(verbose=-1),
    'LGBMRegressor': lgb(verbose=-1),
    'CatBoostRegressor': cat(silent=True),
}

for i, (datasets_name, (x, y)) in enumerate(datasets.items()):
    print(f'\n{datasets_name} dataset start...........................................')
    
    num_classes = len(np.unique(y))
    is_multi = num_classes > 2
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3112, shuffle=True
    )
    
    best_r2 = 0
    best_scaler = None
    best_model = None

    for j, scaler_name in enumerate(scalers):
        for h, model_name in enumerate(models):
            scaler = scalers[scaler_name]
            model = models[model_name]
                
            model = make_pipeline(PCA(n_components=8), scaler, model)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            print(f'model : {model_name:<16}\tscaler : {scaler_name:<9}\tr2 : {r2:.4f}')
            
            if r2 > best_r2:
                best_r2 = r2
                best_scaler = scaler_name
                best_model = model_name
    print(f'\n########## {datasets_name} best combo ##########')
    print(f'best model : {best_model}\nbest scaler : {best_scaler}\nbest r2 : {best_r2:.4f}')