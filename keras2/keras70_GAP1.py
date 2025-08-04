import numpy as np
import pandas as pd
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), strides=1, input_shape=(10, 10, 1)))
                                                # strides = 보폭, default는 1
                                                # kernel_size로 자르는 간격을 설정해주는 것
model.add(Conv2D(filters=50, kernel_size=(2,2)))
model.add(Conv2D(filters=30, kernel_size=(2,2), padding='same'))

model.add(Flatten())      
# model.add(GlobalAveragePooling2D())

model.add(Dense(units=10, activation='softmax'))
model.summary()

# Model: "sequential" <- Flatten 썼을 때
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 9, 9, 100)         500       
#  conv2d_1 (Conv2D)           (None, 8, 8, 50)          20050     
#  conv2d_2 (Conv2D)           (None, 8, 8, 30)          6030      
#  flatten (Flatten)           (None, 1920)              0         
#  dense (Dense)               (None, 10)                19210     
# =================================================================
# Total params: 31,390
# Trainable params: 31,390
# Non-trainable params: 0
# _________________________________________________________________


# Model: "sequential" <- GlobalAveragePooling2D 썼을 때
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 9, 9, 100)         500       
#  conv2d_1 (Conv2D)           (None, 8, 8, 50)          20050     
#  conv2d_2 (Conv2D)           (None, 8, 8, 30)          6030      
#  global_average_pooling2d (G  (None, 30)               0         
#  lobalAveragePooling2D)                                          
#  dense (Dense)               (None, 10)                310     
# =================================================================
# Total params: 26,890
# Trainable params: 26,890
# Non-trainable params: 0
# _________________________________________________________________