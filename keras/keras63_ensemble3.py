import time
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T
# (100, 2) / 삼성전가 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
# (100, 3) / 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T
y1 = np.array(range(2001,2101))
# (100,) / 한강의 화씨 온도
y2 = np.array(range(13001,13101))
# (100,) / 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,\
    y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets,
    y1, y2, random_state=55, test_size=0.3
)   # \ 줄바꿈

print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape)    # (70, 4) (30, 4)
print(y1_train.shape, y1_test.shape)      # (70,) (30,)
print(y2_train.shape, y2_test.shape)      # (70,) (30,)

#2-1. 모델구성
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
dense5 = Dense(50, activation='relu', name='ibm5')(dense4)
dense6 = Dense(60, activation='relu', name='ibm6')(dense5)
dense7 = Dense(70, activation='relu', name='ibm7')(dense6)
output1 = Dense(80, activation='relu', name='ibm8')(dense7)
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

#2-2. 모델구성
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(80, activation='relu', name='ibm22')(dense21)
dense23 = Dense(40, activation='relu', name='ibm23')(dense22)
output2 = Dense(20, activation='relu', name='ibm24')(dense23)
# model21 = Model(inputs=input2, outputs=output2)

#2-3. 모델구성
input3 = Input(shape=(4,))
dense31 = Dense(64, activation='relu', name='ibm31')(input3)
dense32 = Dense(32, activation='relu', name='ibm32')(dense31)
dense33 = Dense(16, activation='relu', name='ibm33')(dense32)
output3 = Dense(8, activation='relu', name='ibm34')(dense33)

#2-3. 모델병합
from keras.layers import Concatenate
# merge1 = concatenate([output1, output2, output3], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output2, output3]) # 대문자와 소문자의 차이

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
model = Model(inputs=[input1, input2, input3], outputs=[last_output3, last_output4])


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

# path = './_save/keras63/'
# date = datetime.datetime.now().strftime('%m%d_%H_%M')
# filename = '{epoch:03d}_{val_loss:.3f}.hdf5'
# filepath = ''.join([path, 'k63_2_', date, '_', filename])
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=2
# )

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=5000,
          batch_size=64, verbose=2, callbacks=[es]
)

#4. 평가, 예측
# y_pred = 2101~2106

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print(results)
y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print(y_pred)

