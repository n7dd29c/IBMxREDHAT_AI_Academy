#가중치 연산이나 numpy 연산엔 joblib이 우수하다

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)  # scikit-learn 데이터 전용, x y 만 따로 빠짐
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, stratify=y
)

parameters = {
    'n_estimaters' : 1000,
    'learning_rate' : 0.3,
    'max_depth' : 3,
    'gemma' : 1,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 337,
    # 'verbose' : 0,
}

#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(**parameters)
model.fit(x_train, y_train, verbose=10, )

results = model.score(x_test, y_test)
print(results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print(acc)

path = './_save/m01_job/'
# import joblib
# joblib.dump(model, path + 'm01_joblib_save.joblib')

model.save_model(path + 'm03_xgb_save.dat')