import numpy as np
import tensorflow as tf
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
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

vgg16.trainable = True              # 가중치 동결

model = Sequential([
    vgg16,
    Flatten(),
    Dense(100),
    Dense(100),
    Dense(10, activation='softmax'),
])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.2, verbose=2)
end_time = time.time()-start_time

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
print(end_time)

# 0.9577142596244812
# 0.6664999723434448
# 408.9307882785797

# loss : 1.6797337532043457
# acc : 0.8256000280380249
# 1205.495371580124

# 실습
# 비교할데이터
# 1. 이전 내가 한 최상의 결과가
# 2. 가중치를 동결하지 않고
# 3. 가중치를 동결하고
# 4. 시간까지 비교한다
# 추가로 Flatten, GAP 비교

######## 실습할 것 ########
# cifar10
# cifar100
# horse
# rps
# cat dog
# men women

