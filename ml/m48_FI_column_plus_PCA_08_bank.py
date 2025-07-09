from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import random
import numpy as np
import pandas as pd

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 문자 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv.shape)  # (165034, 11)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)

#2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print('=============', model.__class__.__name__, '=============')
print('ACC origin : ', model.score(x_test, y_test))    # ACC origin :  0.8650286302905444
print(model.feature_importances_)
        
# [0.47785544 0.06601944 0.04326427 0.02447754 0.02543438 0.1539704 0.09944526 0.10953324]
print(np.percentile(model.feature_importances_, 25))
# 0.0388068

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile)) # <class 'numpy.float32'>

col_name = []
# ['AveBedrms', 'Population']

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
print(col_name)

x_f = pd.DataFrame(x, columns=x.columns)
x1 = x_f.drop(columns=col_name)
x2 = x_f[col_name]

x1_train, x1_test = train_test_split(
    x1, test_size=0.2, random_state=seed,
)

model.fit(x1_train, y_train)
print('ACC drop : ', model.score(x1_test, y_test))    # ACC drop :  0.8646953676492866

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size=0.2, random_state=seed,
)
print(x1_train.shape, x1_test.shape)    # (16512, 6) (4128, 6)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape)    # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)    # (16512, 7) (4128, 7)

model.fit(x_train, y_train)
print('ACC PCA : ', model.score(x_test, y_test))    # ACC PCA :  0.8641803253855243