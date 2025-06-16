import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

train_path = './_data/kaggle/cat_dog/train2/'
test_path = './_data/kaggle/cat_dog/test2/'

xy_train = train_data_gen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=333,
)

xy_test = test_data_gen.flow_from_directory(
    test_path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    seed=333,
)

print(xy_train[0][0].shape) # (100, 200, 200, 3)
print(xy_train[0][1].shape) # (100,)
print(len(xy_train))        # 250

##### 모든 수치화된 batch데이터를 하나로 합치기 #####
all_x_train = []
all_y_train = []
all_x_sub = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x_train.append(x_batch)
    all_y_train.append(y_batch)
    
for i in range(len(xy_test)):
    x_batch, y_batch = xy_test[i]
    all_x_sub.append(x_batch)

x_tr = np.concatenate(all_x_train, axis=0)
y_tr = np.concatenate(all_y_train, axis=0)
x_sub = np.concatenate(all_x_sub, axis=0)

print('x.shape', x_tr.shape)   # x.shape (25000, 200, 200, 3)
print('y.shape', y_tr.shape)   # y.shape (25000,)

np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + 'keras44_01_x_train_catdog.npy', arr=x_tr)
np.save(np_path + 'keras44_01_y_train_catdog.npy', arr=y_tr)
np.save(np_path + 'keras44_01_x_sub_catdog.npy', arr=x_sub)