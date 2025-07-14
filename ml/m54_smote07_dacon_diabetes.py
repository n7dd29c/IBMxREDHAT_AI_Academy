
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
path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())


#  shape 확인
print(train_csv.shape)          # (652, 9)
print(test_csv.shape)           # (116, 8)

# 컬럼확인
print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
print(test_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age'],

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())   # 결측치 없음

#train_csv = train_csv.dropna()

###### x와 y 분리 ####
x = train_csv.drop(['Outcome'], axis=1) # (652, 9)
y = train_csv['Outcome']                # (652,)
print("ㅡㅡㅡㅡㅡㅡㅡ")
print(y.shape) 

# 결측치 처리 
# 특정 생물학적 데이터는 0이 될 수없음. 이 train 데이터는 결측치를 0으로 세팅해놔서 0을 nan으로 대체하고 결측치처리해야함
# 여기서 결측치 처리하는 이유는 Outcome(이진분류정답컬럼)에 있는 0을 nan처리하면 안되기때문
# 여기서 결측치 처리할때 dropna를 쓰면 안되는 이유 : 여기서 dropna를 하면 정답지(y)랑 행 갯수가 달라지고 학습-정답 매칭이 안되어서 제대로 학습을 할 수 없다.
x = x.replace(0, np.nan)    
#x = x.fillna(x.mean())
x = x.fillna(x.median())

# 데이터 불균형 확인
print(pd.value_counts(y))
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
print(np.unique(y, return_counts=True))     # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

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

print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)

############### SMOTE ###############
#꼭 split 하고 SMOTE 쓰기
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('sklearn version: ', sk.__version__) #1.6.1
import imblearn
print('imblearn version: ', imblearn.__version__) #0.12.4

smote = SMOTE(random_state=seed,       
              k_neighbors=5,                  #디폴트
            #   sampling_strategy='auto',     #디폴트
            #   sampling_strategy=0.75,       #최대값의 75% 지정
              sampling_strategy={0:1000, 1:1000}  #직접 지정
            #   n_jobs=-1, #0.13에서는 삭제됨. 그냥 포함됨
              )
x_train , y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([1000, 1000]))

#2. 모델
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer = 'adam', metrics=['acc']) 
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

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
# accuracy_score:  0.7862595419847328
# f1_score:  0.6666666666666666

#2. 2번에서 SMOTE 적용
# accuracy_score:  0.7557251908396947
# f1_score:  0.6190476190476191

#3. 재현's SMOTE
# accuracy_score:  0.7633587786259542
# f1_score:  0.6265060240963856