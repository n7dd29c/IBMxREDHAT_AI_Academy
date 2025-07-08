from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import random
import numpy as np

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
data1 = load_breast_cancer()
data2 = load_digits()
data3 = load_wine()

datasets = [data1, data2, data3]
dataset_name = ['cancer', 'digits', 'wine']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed,
        stratify=y
    )

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print(f'\n################################# {dataset_name[i]} #################################')
    
    for model in models:
        model.fit(x_train, y_train)
        print('\n======================', model.__class__.__name__, '======================')
        print('acc : ', model.score(x_test, y_test))
        print(model.feature_importances_)
    


