# 참고 : Pseudo Labeling

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,\
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=3112, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbose=0)
hgb = HistGradientBoostingClassifier()

models = [xgb, rf, cat, lg, hgb]
train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append((y_test_pred))
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))
    
# XGBClassifier ACC : 0.9649
# RandomForestClassifier ACC : 0.9737
# CatBoostClassifier ACC : 0.9825
# LGBMClassifier ACC : 0.9737
# HistGradientBoostingClassifier ACC : 0.9649
    
x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (16512, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape)     # (4128, 4)

#2-2. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print(score2)

# CatBoostClassifier
# 0.9824561403508771