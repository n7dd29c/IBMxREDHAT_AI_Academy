import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

#1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
# np.load(np_path + 'keras44_01_x_train.npy', arr=x)
# np.load(np_path + 'keras44_01_y_train.npy', arr=y)

start = time.time()
x_train = np.load(np_path + 'keras44_01_x_train.npy')
y_train = np.load(np_path + 'keras44_01_y_train.npy')
end = time.time()

print(x_train)
print(y_train[:20])
print(x_train.shape, y_train.shape)
print(round(end-start, 2), '초')  # 34.01 초