# 실습
# cancer, dacon_diabetes, bank, wine, digits / 5개의 데이터셋을
# pipeline을 통해 여러 모델과 여러 스케일러를 비교하고
# 제일 성능이 좋았을 때 어떤 모델과 스케일러를 썼는지, 점수 출력

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier as xgb
from lightgbm import LGBMClassifier as lgb
from catboost import CatBoostClassifier as cat

import warnings
warnings.filterwarnings('ignore')

#1. 데이터

############################################################ digits
x1, y1 = load_digits(return_X_y=True)
print(x1.shape, y1.shape) # (1797, 64) (1797,)

############################################################ cancer
x2, y2 = load_breast_cancer(return_X_y=True)
print(x2.shape, y2.shape) # (569, 30) (569,)

############################################################ wine
x3, y3 = load_wine(return_X_y=True)
print(x3.shape, y3.shape) # (178, 13) (178,)

############################################################ bank
path = './_data/kaggle/bank/'
train_bank = pd.read_csv(path + 'train.csv', index_col=0)
print(train_bank)        # [165034 rows x 13 columns]
test_bank = pd.read_csv(path + 'test.csv', index_col=0)
print(test_bank)         # [110023 rows x 12 columns]
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_bank['Geography'] = le_geo.fit_transform(train_bank['Geography'])
train_bank['Gender'] = le_gen.fit_transform(train_bank['Gender'])

test_bank['Geography'] = le_geo.transform(test_bank['Geography'])
test_bank['Gender'] = le_gen.transform(test_bank['Gender'])

train_bank = train_bank.drop(["CustomerId", "Surname"], axis=1)
test_bank = test_bank.drop(["CustomerId", "Surname"], axis=1)
print(test_bank.shape)   # (110023, 10)

x4 = train_bank.drop(['Exited'], axis=1)  
print(x4.shape)  # (165034, 10)
y4 = train_bank['Exited']
print(y4.shape)  # (165034,)

############################################################ diabetes
path = './_data/dacon/diabetes/'
train_diab = pd.read_csv(path + 'train.csv', index_col=0)
print(train_diab)        # [652 rows x 9 columns]
test_diab = pd.read_csv(path + 'test.csv', index_col=0)
print(test_diab)         # [116 rows x 8 columns]
x5 = train_diab.drop(['Outcome'], axis=1) # (652, 9)
y5 = train_diab['Outcome']     

datasets = {
    'digits':   (x1, y1),
    'cancer':   (x2, y2),
    'wine':     (x3, y3),
    'diabetes': (x5, y5),
    'bank':     (x4, y4),
}

scalers = {
    'MinMax':    MinMaxScaler(),
    'Standard':  StandardScaler(),
    'Robust':    RobustScaler(),
    'MaxAbs':    MaxAbsScaler(),
}

models = {
    'xgb': xgb(verbose=-1),
    'lgb': lgb(verbose=-1),
    'cat': cat(silent=True),
}

for i, (datasets_name, (x, y)) in enumerate(datasets.items()):
    print(f'\n{datasets_name} dataset start..................................')
    
    num_classes = len(np.unique(y))
    is_multi = num_classes > 2
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3112, shuffle=True, stratify=y
    )
    
    best_acc = 0
    best_scaler = None
    best_model = None

    for j, scaler_name in enumerate(scalers):
        for h, model_name in enumerate(models):
            scaler = scalers[scaler_name]
            model = models[model_name]
            
            if model_name == 'xgb':
                if is_multi:
                    model = xgb(objective='multi:softmax',
                                num_class=num_classes,
                                eval_metric='mlogloss',
                                verbosity=0)
                else:
                    model = xgb(objective='binary:logistic',
                                eval_metric='logloss',
                                verbosity=0)
            elif model_name == 'lgb':
                model = lgb(objective='multiclass' if is_multi else 'binary',
                            num_class=num_classes if is_multi else None,
                            verbose=-1)
            elif model_name == 'cat':
                model = cat(verbose=0)
                
            model = make_pipeline(PCA(n_components=8), scaler, model)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'model : {model_name:<6}\tscaler : {scaler_name:<9}\tacc : {acc:.4f}')
            
            if acc > best_acc:
                best_acc = acc
                best_scaler = scaler_name
                best_model = model_name
    print(f'{datasets_name} best combo')
    print(f'best model : {best_model}\nbest scaler : {best_scaler}\nbest acc : {best_acc:.4f}')