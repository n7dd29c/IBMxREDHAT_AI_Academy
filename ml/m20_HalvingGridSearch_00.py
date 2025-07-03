from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import time
import joblib

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=55
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=503)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]

#2. 모델
xgb = XGBClassifier()
model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
                     verbose=2,
                     n_jobs=-1,
                     refit=True,        # 1번
                     random_state=98,
                     factor=3,          # 데이터 자를 비율, default=3
                                        # 배율 : min_resources * factor
                                        #          n_candinates
                     min_resources=10,  # 학습할 최소 샘플 수 (1 iter 당)
                     max_resources=120, # 학습할 최대 샘플 수 (행의 갯수, n_samples)
                     aggressive_elimination=True,
                     )

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('- accuracy_score : ', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('- best_acc_score : ', accuracy_score(y_test, y_pred_best))   # 마지막 최적값으로 예측

print('- 걸린시간 : ', round(end, 2), '초\n\n')

# - 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=100,
#               n_jobs=None, num_parallel_tree=None, ...)
# - 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
# - best_score :  0.95
# - mode.score :  0.9666666666666667
# - accuracy_score :  0.9666666666666667
# - best_acc_score :  0.9666666666666667
# - 걸린시간 :  3.46 초

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values(                      # cross validation에 대한 결과값
                                                    'rank_test_score',  # rank test score 기준으로
                                                    ascending=True))    # 오름차순

print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_learning_rate', 'param_max_depth', 'param_n_estimators',
#        'param_min_child_weight', 'params', 'split0_test_score',
#        'split1_test_score', 'split2_test_score', 'split3_test_score',
#        'split4_test_score', 'mean_test_score', 'std_test_score',
#        'rank_test_score'],
#       dtype='object')