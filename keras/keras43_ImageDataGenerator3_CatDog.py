import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

#1. 데이터
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
)

test_data_gen = ImageDataGenerator(
    rescale=1./255,
)

train_path = './Study25/_data/kaggle/cat_dog/train2/'
test_path = './Study25/_data/kaggle/cat_dog/test2/'

xy_train = train_data_gen.flow_from_directory(
    train_path,
    target_size=(400, 400),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=333,
)

print(xy_train[0][0].shape) # (100, 200, 200, 3)
print(xy_train[0][1].shape) # (100,)
print(len(xy_train))        # 250

##### 모든 수치화된 batch데이터를 하나로 합치기 #####
all_x = []
all_y = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    
# print(all_x)

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

print('x.shape', x.shape)   # x.shape (25000, 200, 200, 3)
print('y.shape', y.shape)   # y.shape (25000,)

exit()  
# xy_test = test_data_gen.flow_from_directory(
#     test_path,
#     target_size=(200, 200),
#     batch_size=100,
#     class_mode='binary',
#     color_mode='rgb',
# )

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


#2. 모델구성
model = Sequential()
model.add(Conv2D(256, 2, 1, input_shape=(400, 400, 3)))
model.add(Conv2D(256, 2, 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(64, 2, 1, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train, epochs=3000, validation_split=0.2,
          callbacks=es, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print('runtime : ', end_time-start_time)

import matplotlib.pyplot as plt
plt.imshow(x_test[1], 'gray')
plt.show()