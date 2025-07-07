from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#train_test_split 후 scaling 후 PCA


for i in range(x.shape[1]):
    pca = PCA(n_components=i+1)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    #2. 모델
    model = RandomForestClassifier(random_state=333)
    
    #3. 훈련
    model.fit(x_train_pca, y_train)
    
    #4. 평가
    results = model.score(x_test_pca, y_test)
    print(x_test_pca.shape, '의 score: ', results)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
print(evr)                          # [0.73616646 0.22089149 0.03787614 0.00506591]
print(sum(evr))                     # 0.9999999999999998

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()