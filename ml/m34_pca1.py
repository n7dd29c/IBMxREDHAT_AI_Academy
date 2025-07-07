from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)

# PCA는 Scaler를 여기에 보통 쓴다
scaler = StandardScaler()
x = scaler.fit_transform(x)

# PCA는 비지도학습, 보통 y가 없으면 비지도학습이다
pca = PCA(n_components=3)
x = pca.fit_transform(x)
print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, stratify=y
)

#2. 모델
model = RandomForestClassifier(random_state=333)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print(x.shape, '|', results)