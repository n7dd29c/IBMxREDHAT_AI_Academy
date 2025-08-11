# california, diabetes : XGB
# cancer, digits : LGBM

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=3112,
)

parameter = [
    {'xgb__n_estimators': [100,200], 'xgb__max_depth': [5,6,10], 'xgb__min_samples_leaf': [3,10]},
    {'xgb__max_depth':[6,8,10,12], 'xgb__min_samples_leaf': [3,5,7,10]},
    {'xgb__min_samples_leaf': [3,5,7,9], 'xgb__min_samples_split': [2,3,5,10]},
    {'xgb__min_samples_split': [2,3,5,8]},
]

# #2. 모델
pipe = Pipeline([('std', StandardScaler()), ('xgb', XGBRegressor())])
model = GridSearchCV(pipe, parameter, cv=5, verbose=0, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('score :', results)

y_pred = model.predict(x_test)
acc = r2_score(y_test, y_pred)
print('acc :', acc)

# score : 0.8414848512616514
# acc : 0.8414848512616514