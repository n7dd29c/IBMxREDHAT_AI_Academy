import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd # pandas를 import하여 컬럼 이름을 사용하면 좋습니다.

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
# x를 pandas DataFrame으로 변환하여 컬럼 이름을 활용 (옵션)
feature_names = fetch_california_housing().feature_names
x_df = pd.DataFrame(x, columns=feature_names)

# 각 컬럼별 이상치 경계를 계산하는 함수
def outlier(data):
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    # 이상치의 위치는 여기서는 반환하지 않고, 경계값만 반환
    return iqr, lower_bound, upper_bound

# 컬럼별 정보를 저장할 리스트
iqrs = []
lowers = []
uppers = []

# 각 컬럼에 대해 outlier_per_column 함수 적용
for i in range(x_df.shape[1]):
    col_data = x_df.iloc[:, i] # i번째 컬럼 데이터
    iqr, low, upp = outlier(col_data)
    iqrs.append(iqr)
    lowers.append(low)
    uppers.append(upp)

# print('컬럼별 IQR :', iqrs)
# print('컬럼별 Lower Bounds :', lowers)
# print('컬럼별 Upper Bounds :', uppers)

# plt.show()를 한 창에 8개의 표로 나누어 그리기
# 2행 4열 그리드로 서브플롯을 생성
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10)) # figsize 조정 가능
axes = axes.flatten() # 2D array of axes를 1D array로 평탄화하여 반복문에서 쉽게 접근

for i, col_name in enumerate(x_df.columns):
    ax = axes[i] # 현재 서브플롯
    ax.boxplot(x_df[col_name]) # 해당 컬럼의 박스플롯
    
    # 해당 컬럼에 대한 상/하한선 표시
    ax.axhline(uppers[i], color='pink', label='upper bound')
    ax.axhline(lowers[i], color='pink', label='lower bound')
    
    ax.set_title(f'{col_name}') # 컬럼 이름으로 제목 설정
    ax.legend(fontsize=8, loc='upper right') # 범례 크기 조정

plt.tight_layout() # 서브플롯 간의 간격 자동 조절
# plt.show()

# x[0,0] x[0,1] x[0,2]
for i in range(x.shape[1]):
    for j in range(len(x)):
        if x[j,i] > uppers[i]:
            x[j,i] = uppers[i]
        elif x[j,i] < lowers[i]:
            x[j,i] = lowers[i]
        else:
            continue

plt.figure(figsize=(9,9))
plt.boxplot(x)
plt.show()

# exit()
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x_df, y, cv=kfold) # fit까지 포함
print('acc : ', scores, '\n평균 acc : ', round(np.mean([scores]), 3))