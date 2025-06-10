import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = Sequential()
model.add(Conv2D(filters=5, kernel_size=(2,2), strides=2, input_shape=(10, 10, 1)))
                                                # strides = 보폭, default는 1
                                                # kernel_size로 자르는 간격을 설정해주는 것
model.add(Conv2D(filters=4, kernel_size=(2,2)))
model.add(Flatten())      
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 5, 5, 5)           25         10을 2*2만큼 2간격으로 자름
#  conv2d_1 (Conv2D)           (None, 4, 4, 4)           84         strides 명시를 안해서 자동으로 1
#  flatten (Flatten)           (None, 64)                0
# =================================================================