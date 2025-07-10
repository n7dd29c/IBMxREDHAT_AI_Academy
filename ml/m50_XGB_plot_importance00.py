# m44_0 카피

#42_1 카피

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

seed = 1233
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

#2. 모델
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

# plot_feature_importance_datasets(model)
# plt.show()


model4.fit(x_train, y_train)
# print("=================", model4.__class__.__name__, "=================" )
# print('acc :', model4.score(x_test, y_test))
# print(model4.feature_importances_)
# plot_feature_importance_datasets(model4)


from xgboost.plotting import plot_importance
# plot_importance(model4, importance_type='weight') 
plot_importance(model4, importance_type='gain',
                title = 'feature importance [gain]')  # 각 feature가 성능 개선에 얼마나 기여했는지
# plot_importance(model4, importance_type='cover') # split된 sample 수의 평균
plt.show()

'''
weight : 얼마나 자주 split했냐. 통상 frequency 
gain : split이 모델의 성능을 얼마나 개선했냐 // 가장 많이 쓰임
cover : split하기 위한 sample 수. 별로.
'''