# california, diabetes : XGB
# cancer, digits : LGBM

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=3112,
)

parameter = [
    {'lgb__n_estimators': [100,200], 'lgb__max_depth': [5,6,10], 'lgb__min_samples_leaf': [3,10]},
    {'lgb__max_depth':[6,8,10,12], 'lgb__min_samples_leaf': [3,5,7,10]},
    {'lgb__min_samples_leaf': [3,5,7,9], 'lgb__min_samples_split': [2,3,5,10]},
    {'lgb__min_samples_split': [2,3,5,8]},
]

# #2. 모델
pipe = Pipeline([('std', StandardScaler()), ('lgb', LGBMClassifier())])
model = GridSearchCV(pipe, parameter, cv=5, verbose=0, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('score :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

# score : 0.38520249164165343
# acc : 0.38520249164165343