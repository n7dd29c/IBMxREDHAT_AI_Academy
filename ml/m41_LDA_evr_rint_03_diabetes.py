from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets.target
y_origin = y.copy()

print(x.shape, y.shape)     #(442, 10) (442,)
print(y)                    # int형 데이터가 아님
y = np.rint(y).astype(int)  # int형 데이터로 변경
print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test, y_train_o, y_test_o = train_test_split(
    x, y, y_origin, test_size=0.2, random_state=333,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#region PCA 방식
# pca = PCA(n_components=10)    # default : n_components=max
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# pca_evr = pca.explained_variance_ratio_
# print(np.cumsum(pca_evr))
# [0.40986356 0.55677371 0.67171464 0.76820003 0.83381774 0.89297489
#  0.94489495 0.99120464 0.99916639 1.        ]
#endregion

#region LDA 방식
lda = LinearDiscriminantAnalysis(n_components=10)    # y라벨의 갯수 - 1
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)
lda_evr = lda.explained_variance_ratio_
print(lda_evr)
print(np.cumsum(lda_evr))
# # [0.26265604 0.38673594 0.49780081 0.59569547 0.6880911  0.76913686
# #  0.83723877 0.89979303 0.95625834 1.        ]
#endregion

#2. 모델
model = RandomForestRegressor(random_state=333)

#3. 훈련
model.fit(x_train, y_train_o)

#4. 평가
results = model.score(x_test, y_test_o)
print(' score: ', results)

# PCA : score:  0.3876834482836953
# LDA : score:  0.33279932066300566