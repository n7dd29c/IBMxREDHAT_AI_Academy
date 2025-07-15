from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x = np.log1p(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)

#2. 모델
model = RandomForestRegressor(random_state=seed,)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
score = model.score(x_train, y_train)
print(score)

y_pred = model.predict(x_test)
print(np.mean(y_pred))

# log 변환 x
# 0.9733420438046618
# 2.0483637669089143

# log 변환
# 0.9632822832211341
# 2.056648345445736