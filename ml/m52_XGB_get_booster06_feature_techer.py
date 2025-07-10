#49_06 카피
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor

seed = 123
random.seed(seed)
np.random.seed(seed)

delete_columns = []
max_acc = 0

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
feature_names = datasets.feature_names
print(x.shape, y.shape) # (569, 30) (569,)
print(feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'logloss',
    data_name = 'validation_0', 
    # save_best = True,
)

model = XGBClassifier(random_state=seed,
                      n_estimators = 500,
                      max_depth = 0,
                      gamma = 0,
                      min_child_weight = 0,
                      subsample = 0.4,
                      reg_alpha = 0,
                      reg_lambda = 1,
                      eval_metric = 'logloss',   # 다중분류 : mlogloss, merror / 이진분류 : logloss, error / 회귀 : rmse, mae, rmsle
                                                    # 2.1.1 버전 이후로 fit에서 모델로 위치이동
                    #   callbacks = [es]
    )

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )

print('acc :', model.score(x_test, y_test))
# print(model.feature_importances_)

# aaa = model.get_booster().get_score(importance_type='weight') # split 빈도수 개념
# {'f0': 5.0, 'f1': 27.0, 'f3': 4.0, 'f4': 10.0, 'f5': 4.0, 'f6': 5.0, 
#  'f7': 20.0, 'f9': 3.0, 'f10': 3.0, 'f12': 2.0, 'f13': 19.0, 'f14': 3.0, 
#  'f15': 12.0, 'f17': 3.0, 'f18': 6.0, ' f19': 3.0, 'f20': 8.0, 'f21': 33.0, 
#  'f22': 10.0, 'f23': 24.0, 'f24': 15.0, 'f25': 1.0, 'f26': 13.0, 
#  'f27': 13.0, 'f28': 9.0, 'f29': 2.0}

score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)
total = sum(score_dict.values())
print(total)    

# threshold 정규화 (모든 중요도 값을 합으로 나눔)
score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
print(score_list)
print(len(score_list))

# 정규화된 thresholds 정렬
thresholds = np.sort(list(score_list))  # 오름차순
print("정규화된 thresholds:", thresholds)

############################### 컬럼명 매칭 ###############################

score_df = pd.DataFrame({
    'feature' : feature_names,
    'gain' : list(score_dict.values())
}).sort_values(by='gain', ascending=True)
print(score_df)

from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False) 
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다. (기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때. fit 호출하지 않음.
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    mask = ~selection.get_support()
    
    not_select_features = [feature_names[j] for j, selected in enumerate(mask) if not selected]
    
    select_model = XGBClassifier(random_state=seed,
                                 n_estimators = 500,
                                 max_depth = 0,
                                 gamma = 0,
                                 min_child_weight = 0,
                                 subsample = 0.4,
                                 reg_alpha = 0,
                                 reg_lambda = 1,
                                 eval_metric = 'logloss',
                                 )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = False,
          )
        
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, ACC=%.4f%%'%(i, select_x_train.shape[1], score*100))
    print('삭제된 컬럼 : ', not_select_features)
    print('-------------------------------------------------')
    
    if max_acc<=score:
        max_acc = score
        delete_columns = list(not_select_features)
        nn = select_x_train.shape[1]
        threch = i
        
print(f'Max ACC = {np.round(max_acc*100, 4)}%')
print('gain : ', threch)
print('삭제 할 컬럼 : ', delete_columns)
print('삭제 할 컬럼 수 : ', nn)

