import numpy as np
aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]

# 공분산과 평균을 이용해서 데이터를 타원형태의 군집으로 그린 후,
# 'Mahalanobis 거리'를 구해서 이상치를 찾는다