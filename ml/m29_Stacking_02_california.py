import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=3113, #stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)
lg = LGBMRegressor(verbose=0)
hgb = HistGradientBoostingRegressor()

models = [xgb, rf, cat, lg, hgb]
train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append((y_test_pred))
    
    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} R2 : {1:.4f}'.format(class_name, score))
    
# XGBRegressor R2 : 0.8441
# RandomForestRegressor R2 : 0.8232
# CatBoostRegressor R2 : 0.8582
# LGBMRegressor R2 : 0.8462
# HistGradientBoostingRegressor R2 : 0.8419
    
x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (16512, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape)     # (4128, 4)

#2-2. 모델
model2 = XGBRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print(score2)

# CatBoostRegressor
# 0.8071525236499778

# LGBMRegressor
# 0.8055004654781823

# XGBRegressor
# 0.8087351843286339

# RandomForestRegressor
# 0.7998943771170727

# XGBRegressor - MinMaxScaler
# 0.8082102194176537