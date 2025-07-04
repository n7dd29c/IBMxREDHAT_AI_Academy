from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random

seed = 333
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = fetch_covtype(return_X_y=True)
y = y -1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

parameters = {
    # XGBoost Parameters
    'xgb_n_estimators': (100, 500),
    'xgb_learning_rate': (0.01, 0.2),
    'xgb_max_depth': (3, 8),
    'xgb_subsample': (0.6, 1.0),
    'xgb_colsample_bytree': (0.6, 1.0),
    'xgb_gamma': (0, 5),

    # LightGBM Parameters
    'lgbm_n_estimators': (100, 500),
    'lgbm_learning_rate': (0.01, 0.2),
    'lgbm_max_depth': (3, 8), # LightGBM은 더 깊게도 가능하지만, 앙상블이므로 적당히 제한
    'lgbm_num_leaves': (20, 60), # max_depth와 연관
    'lgbm_min_child_samples': (10, 50),
    'lgbm_subsample': (0.6, 1.0),
    'lgbm_colsample_bytree': (0.6, 1.0),

    # CatBoost Parameters (CatBoost는 파라미터 이름이 XGB/LGBM과 다를 수 있음)
    'cat_iterations': (100, 500), # n_estimators와 유사
    'cat_learning_rate': (0.01, 0.2),
    'cat_depth': (3, 8), # max_depth와 유사
    'cat_l2_leaf_reg': (1, 10), # reg_lambda와 유사 (L2 regularization)
    'cat_random_strength': (0, 10), # overfitting 방지
}

def ensemble_objective(xgb_n_estimators, xgb_learning_rate, xgb_max_depth, xgb_subsample, xgb_colsample_bytree, xgb_gamma,
                       lgbm_n_estimators, lgbm_learning_rate, lgbm_max_depth, lgbm_num_leaves, lgbm_min_child_samples,
                       lgbm_subsample, lgbm_colsample_bytree,
                       cat_iterations, cat_learning_rate, cat_depth, cat_l2_leaf_reg, cat_random_strength):
    
    # XGBoost 모델 설정
    xgb_params = {
        'n_estimators': int(round(xgb_n_estimators)),
        'learning_rate': xgb_learning_rate,
        'max_depth': int(round(xgb_max_depth)),
        'subsample': xgb_subsample,
        'colsample_bytree': xgb_colsample_bytree,
        'gamma': xgb_gamma,
        'random_state': seed,
        'verbosity': 0, # 로그 출력 억제
        'n_jobs': -1,
    }
    xgb_model = XGBClassifier(**xgb_params, tree_method='gpu_hist')

    # LightGBM 모델 설정
    lgbm_params = {
        'n_estimators': int(round(lgbm_n_estimators)),
        'learning_rate': lgbm_learning_rate,
        'max_depth': int(round(lgbm_max_depth)),
        'num_leaves': int(round(lgbm_num_leaves)),
        'min_child_samples': int(round(lgbm_min_child_samples)),
        'subsample': lgbm_subsample,
        'colsample_bytree': lgbm_colsample_bytree,
        'random_state': seed,
        'verbosity': -1, # 로그 출력 억제
        'n_jobs': -1,
    }
    lgbm_model = LGBMClassifier(**lgbm_params, tree_method='gpu_hist')
    
    # CatBoost 모델 설정
    cat_params = {
        'iterations': int(round(cat_iterations)), # CatBoost는 iterations 파라미터 사용
        'learning_rate': cat_learning_rate,
        'depth': int(round(cat_depth)), # CatBoost는 depth 파라미터 사용
        'l2_leaf_reg': cat_l2_leaf_reg,
        'random_strength': cat_random_strength,
        'random_seed': seed, # CatBoost는 random_seed 파라미터 사용
        'verbose': False, # 로그 출력 억제
        'thread_count': -1, # n_jobs와 유사
    }
    cat_model = CatBoostClassifier(**cat_params)

    # VotingClassifier 구성
    # Soft Voting을 사용하며, 각 모델은 사전 정의된 파라미터로 초기화됨
    # early_stopping은 개별 모델 학습에 포함되지 않고 최종 앙상블 학습 시에는 고려되지 않음
    ensemble_model = VotingClassifier(
        estimators=[
            ('XGB', xgb_model),
            ('LGBM', lgbm_model),
            ('CAT', cat_model)
        ],
        voting='soft' # 확률 기반 소프트 보팅
    )

    # 앙상블 모델 훈련
    # 개별 모델들은 이미 설정된 n_estimators만큼 학습됨
    ensemble_model.fit(x_train, y_train)

    # 앙상블 모델 예측 및 성능 평가
    y_pred = ensemble_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 베이지안 최적화는 이 accuracy 값을 최대화하도록 파라미터를 탐색
    return accuracy

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = ensemble_objective,
    pbounds=parameters,
    random_state=seed,
    verbose=2
)

n_iter = 100
start = time.time()
optimizer.maximize(init_points=5, n_iter=n_iter)
end = time.time() - start

print(optimizer.max)
print(n_iter, '번 걸린 시간 : ', round(end), '초')

# 0.886517559787613
# 100 번 걸린 시간 :  2847 초