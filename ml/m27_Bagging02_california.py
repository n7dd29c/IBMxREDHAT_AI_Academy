from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random

seed = 55
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeRegressor()

model = BaggingRegressor(DecisionTreeRegressor(),
                         n_estimators=99,
                         n_jobs=-1,
                         bootstrap=True,
                         random_state=333
                         )

# model = RandomForestRegressor(random_state=332)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x_test)

# DecisionTreeRegressor
# 0.5933155301079609

# BaggingRegressor(DecisionTreeRegressor())
# 0.7980567079547046

# RandomForestRegressor
# 0.8002272264062507