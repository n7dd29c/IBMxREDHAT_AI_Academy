# pipeline은 일괄처리와 같다고 보면 됨

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=3112,
    # stratify=y
)

parameter = [
    {'randomforestclassifier_n_estimators': [100,200], 'randomforestclassifier_max_depth': [5,6,10],\
        'randomforestclassifier_min_samples_leaf': [3,10]},
    {'randomforestclassifier_max_depth':[6,8,10,12], 'randomforestclassifier_min_samples_leaf': [3,5,7,10]},
    {'randomforestclassifier_min_samples_leaf': [3,5,7,9], 'randomforestclassifier_min_samples_split': [2,3,5,10]},
    {'randomforestclassifier_min_samples_split': [2,3,5,8]},
]   # Pipeline과 비교하면 앞의 rf__를 쓸 때, make_pipeline은 풀네임을 써줘야한다 

# #2. 모델
pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
model = GridSearchCV(pipe, parameter, cv=5, verbose=0, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('score :', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)

