import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, f1_score
from keras.models import Sequential
from keras.layers import Dense

seed = 44
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset['target']
print(x.shape, y.shape)                     # (178, 13) (178,)

print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71, 48]))

print(pd.value_counts(y))                   # 1    71
                                            # 0    59
                                            # 2    48
                                            # Name: count, dtype: int64
                                            
print(y)

### 데이터 삭제, 라벨이 2인것을 8개만 남기고 다 지우기
x = x[:-40]
y = y[:-40]
print(np.unique(y, return_counts=True))
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=seed, stratify=y
)

###################### SMOTE 적용 ######################
# pip install imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
import imblearn as imb
print('sklearn version : ', sk.__version__)     # 1.6.1
print('imblearn version : ', imb.__version__)   # 0.13.0

# smote = SMOTE(random_state=seed,        # 라벨간의 불균형 해결이라 회귀에선 잘 안씀
#               k_neighbors=5,            # default
#               sampling_strategy='auto', # default
#             #   sampling_strategy=0.75,     # 최대값의 75% 지점
#             #   sampling_strategy={0:50, 2:33}    # 직접지정 : (array([0, 1, 2]), array([50, 53, 33]))
#                                                   # sampling_strategy를 지정 안하면 제일 많은 값으로 맞춰짐
#                                                   # array([50, 53, 33])라면 53에 맞춰짐
#               )

# x_train, y_train = smote.fit_resample(x_train, y_train)

ros = RandomOverSampler(random_state=seed,
                        sampling_strategy={0:5000, 2:5000, 1:5000}
              )

x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2]), array([53, 53, 53]))

#2. 모델
model = Sequential([
    Dense(10, input_shape=(13,)),
    Dense(3, activation='softmax')
])

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', # One-Hot 하지 않은 데이터로는 sparse_ 붙여야함
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
print(y_pred.shape)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape)
print(y_pred)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print('acc : ', acc)
print('f1 : ', f1)

############## 결과 #############

#1. 원데이터 : 
# acc :  0.8222222222222222
# f1 :  0.8203282828282829

#2. 클래스 2를 40개 삭제한 데이터 : 
# acc :  0.8285714285714286
# f1 :  0.578853046594982

#3. SMOTE 적용
# acc :  0.8571428571428571
# f1 :  0.7844155844155845

#4. RandomOverSampler 적용
# acc :  0.9428571428571428
# f1 :  0.9592592592592593