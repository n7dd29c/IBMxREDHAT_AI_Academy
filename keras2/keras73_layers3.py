import numpy as np
import tensorflow as tf
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, AveragePooling2D
from keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, \
    ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

vgg16 = VGG16(
    include_top=False,
    input_shape=(32,32,3),
)

vgg16.trainable = True

model = Sequential([
    vgg16,
    Flatten(),
    Dense(100),
    Dense(100),
    Dense(10, activation='softmax'),
])
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

##### 전체동결1 #####
# model.trainable = False

##### 전체동결2 #####
for layer in model.layers:
    layer.trainable = False # <- 1번과 같음

##### 부분동결 #####
# model.layers[0].trainable = False

import pandas as pd
pd.set_option('max_colwidth', 10)   # None 길이 다 나옴, 10이면 10개만
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Tranable'])
print(results)