import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 0~255 정규화 (scaling)
    horizontal_flip=True,   # 수평 뒤집기, 데이터 증폭 (변환)
    vertical_flip=True,     # 수직 뒤집기, 데이터 증폭 (변환)
    width_shift_range=0.1,  # 수치 평행이동 10%
    height_shift_range=0.1, # 수치 수직이동 10%
    rotation_range=5,       # 5도 회전
    zoom_range=1.2,         # 1.2배 확대
    shear_range=0.7,        # 좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시켜 변환 (이미지 늘리기)
    fill_mode='nearest',    # 빈 공간 근접치로 채움
)

test_datagen = ImageDataGenerator(  # 검증데이터는 증폭, 변환하면 안됨
    rescale=1./255  
)

train_path = './_data/image/brain/train/'
test_path = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    train_path,             # 경로
    target_size=(200, 200), # 사이즈 규격 일치 (큰거는 축소, 작은건 확대)
    batch_size=160,
    class_mode='binary',    # 분류
    color_mode='grayscale',
    shuffle=True,           # default=False    
)

xy_test = test_datagen.flow_from_directory(
    test_path,              # 경로
    target_size=(200, 200), # 사이즈 규격 일치 (큰거는 축소, 작은건 확대)
    batch_size=160,
    class_mode='binary',    # 분류
    color_mode='grayscale',
    # shuffle=True,         # test 데이터는 섞지 않음    
)
# Found 160 images belonging to 2 classes.

print(xy_train)             # <keras.preprocessing.image.DirectoryIterator object at 0x000001C3087F3220>
print(xy_train[0])          # (array([이미지 데이터들]), array([라벨 데이터들]))
print(len(xy_train))        # 16, 길이를 찾는 함수 : len
print(xy_train[0][0].shape) # (10, 200, 200, 1)
print(xy_train[0][1].shape) # (10,)
print(xy_train[0][1])       # [0. 0. 0. 0. 1. 1. 1. 0. 1. 0.]

# print(xy_train[0].shape)    # 'tuple' object has no attribute 'shape'
# print(xy_train[16])         # Asked to retrieve element 16, 15번째까지만 있음
# print(xy_train[0][2])       # tuple index out of range, z 데이터는 없음

print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'> : x numpy 덩어리와 y numpy 덩어리밖에 없는 것
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> : 0번째 배치의 x 데이터
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> : 0번째 배치의 y 데이터