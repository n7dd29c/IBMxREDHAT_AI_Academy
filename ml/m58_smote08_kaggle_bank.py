
import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


from keras.models import Sequential
from keras.layers import Dense, Dropout   
from keras.callbacks import EarlyStopping
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

# 1.데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.head(10)) # 맨 앞 행 10개만 출력
print(test_csv.head(10)) # 맨 앞 행 10개만 출력

# 결측치 확인
print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

#  shape 확인
print(train_csv.shape)          # (165034, 13)
print(test_csv.shape)           # (110023, 12)
print(submission_csv.shape)     # (110023, 2)

# 컬럼명 확인
print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

# 문자 데이터 수치화(인코딩)
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# # 아래 2줄이랑 같다.
# le_geo.fit(train_csv['Geography'])                                    # 'Geography' 컬럼을 기준으로 인코딩한다.
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])     # 적용하고 train_csv['컬럼']에 입력함.
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])

# 테스트 데이터도 수치화해야한다. 위에서 인스턴스가 이미 fit해놨기때문에 transform만 적용한다.
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

print(train_csv.head())

# 변환된 컬럼 데이터의 종류별 갯수 출력(데이터 불균형 확인)
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884

train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)  
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)
exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=seed,
    shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (132027, 10) (33007, 10)
print(y_train.shape, y_test.shape)  # (132027,) (33007,)

print(np.unique(y, return_counts=True))
# (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))

############### SMOTE ###############
#꼭 split 하고 SMOTE 쓰기
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version: ', sk.__version__) #1.6.1
import imblearn
print('imblearn version: ', imblearn.__version__) #0.12.4

ros = RandomOverSampler(random_state=seed,
                        sampling_strategy={0:300000, 1:300000}  #직접 지정
                        )
x_train , y_train = ros.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) 


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', #one-hot 을 안 했기 때무에 categorical_crossentropy가 아님
              optimizer = 'adam', metrics=['acc']) 
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape) # one-hot 형태로 나옴
# y_pred = np.argmax(y_pred, axis=1)
y_pred = (y_pred > 0.5).astype(int).reshape(-1) #이진분류 일 때
print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='binary') # f1은 default로 binary만 받기 때문에 다중일 경우 macro 써줘야됨
print('accuracy_score: ', acc)
print('f1_score: ', f1)

############### 결과 ###############
#1. 변환하지 않은 원 데이터 훈련
# accuracy_score:  0.8615748174629624
# f1_score:  0.6316807738814993

#2. 2번에서 SMOTE 적용
# accuracy_score:  0.8532735480352652
# f1_score:  0.6522082585278276

#3. 재현's SMOTE
# accuracy_score:  0.8457599903050868
# f1_score:  0.6552914889295145

#4. RandomOverSampler 적용
# accuracy_score:  0.8493349895476717
# f1_score:  0.6569635096916604