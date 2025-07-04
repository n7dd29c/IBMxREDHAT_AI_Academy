from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random

seed = 553
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier(random_state=seed, silent=True)
lg = LGBMClassifier(random_state=seed, verbosity=-1)
cat = CatBoostClassifier(random_state=seed, verbose=False)

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    # voting='hard' # default
    voting='soft'
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('Final : ', acc)

# Final :  1.0