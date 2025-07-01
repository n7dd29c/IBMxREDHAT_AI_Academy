import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2. 모델구성
# model = Perceptron()
# model = LogisticRegression()
model = LinearSVC()

#3. 컴파일, 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_pred = model.predict(x_data)
results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_pred)
print('accuracy : ', acc)

# Perceptron
# model.score :  1.0
# accuracy :  1.0

# LogisticRegression
# model.score :  0.75
# accuracy :  0.75

# LinearSVC
# model.score :  1.0
# accuracy :  1.0