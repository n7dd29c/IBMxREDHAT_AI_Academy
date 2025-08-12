# pipeline은 일괄처리와 같다고 보면 됨

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=3112, stratify=y
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# model = RandomForestClassifier()

model = make_pipeline(StandardScaler(), RandomForestClassifier())   # 위의 과정을 한줄로 압축가능
model = make_pipeline(MinMaxScaler(), SVC())                        # 원하는대로 커스텀 가능

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('score :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)
