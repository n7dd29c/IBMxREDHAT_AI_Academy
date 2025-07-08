from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import random
import numpy as np

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    # stratify=y
)

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print('=============', model.__class__.__name__, '=============')
    print('R2 : ', model.score(x_test, y_test))
    print(model.feature_importances_)