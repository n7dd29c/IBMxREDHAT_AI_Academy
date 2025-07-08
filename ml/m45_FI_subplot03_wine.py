from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import random
import numpy as np

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
    stratify=y
)

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

plt.figure(figsize=(15, 10))
plot_num = 1

for model in models:
    model.fit(x_train, y_train)
    print('=============', model.__class__.__name__, '=============')
    print('acc : ', model.score(x_test, y_test))
    print(model.feature_importances_)
    
    plt.subplot(2, 2, plot_num)
    
    n_features = datasets.data.shape[1] # 특성 개수
    feature_names = datasets.feature_names

    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names) # 특성 이름을 y축 레이블로 사용
    plt.xlabel('feature importance')
    plt.ylabel('feature')
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)
    
    plot_num += 1
    
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()