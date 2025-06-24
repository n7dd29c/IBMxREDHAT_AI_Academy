import time
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T
# (100, 2) / 삼성전가 종가, 하이닉스 종가
y1 = np.array(range(2001,2101))
# (100,) / 한강의 화씨 온도
y2 = np.array(range(13001,13101))
# (100,) / 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,\
    y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets,
    y1, y2, random_state=55, 
    test_size=0.3
)   # \ 줄바꿈
    
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
dense5 = Dense(50, activation='relu', name='ibm5')(dense4)

from keras.layers.merge import Concatenate
# merge1 = concatenate([output1, output2, output3], name='mg1')
merge1 = Concatenate(name='mg1')(dense5) # 대문자와 소문자의 차이

merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
middle_output = Dense(1, name='middle')(merge3)

''' 방법 1
# last_output1 = Dense(1, name='last1')(merge3)
# last_output2 = Dense(1, name='last2')(merge3)
# model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])
'''

#2-4. 분리 -> y1
last_output1 = Dense(10, name='last1')(middle_output)
last_output2 = Dense(10, name='last2')(last_output1)
last_output3 = Dense(1, name='last3')(last_output1)

#2-4. 분리 -> y1
last_output4 = Dense(1, name='last4')(middle_output)
model = Model(inputs=input1, outputs=[last_output3, last_output4])