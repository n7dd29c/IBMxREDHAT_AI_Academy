from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)

x = x.reshape(-1, x.shape[1]*x.shape[2])

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

print('1 이상 : ', np.argmax(evr_cumsum>=1.0)+1)
print('0.999이상 : ', np.argmax(evr_cumsum>=0.999)+1)
print('0.99이상 : ', np.argmax(evr_cumsum>=0.99)+1)
print('0.95이상 : ', np.argmax(evr_cumsum>=0.95)+1)

#1. 1.0 일때 몇개인지
#2. 0.999 이상 몇개인지
#3. 0.99 이상 몇개인지
#4. 0.95 이상 몇개인지