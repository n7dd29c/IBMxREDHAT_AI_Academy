import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
x = np.load(np_path + 'keras46_01_x_train_horse.npy')
y = np.load(np_path + 'keras46_01_y_train_horse.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, 
)

#2. 모델구성

#3. 컴파일, 훈련
model_path = './_save/keras45/'
model = load_model(model_path + 'k46_horse_0010_0.9697.hdf5')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
path = './_data/image/me/'
img = np.load(path + 'keras47_haerin.npy')
y_pred = model.predict(img)
# y_pred = model.predict(img)

if y_pred[0][0] < 0.5:
    print('말')
else:
    print('사람')