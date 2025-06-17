from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # (60000, 28, 28)

print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

augment_size = 100          # 증가시킬 사이즈 
aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)  # tile은 numpy data의 단순 복붙
print(aaa.shape)            # (100, 28, 28, 1), augment size 만큼 증가

datagen = ImageDataGenerator(   # 인스턴스화, 실행 할 준비
    rescale=1./255,         # 0~255 정규화 (scaling)
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,  # 수치 평행이동 10%
    height_shift_range=0.1,
    # zoom_range=1.2,
    rotation_range=15,      # 5도 회전
    fill_mode='nearest',    # 빈 공간 근접치로 채움
)

xy_data = datagen.flow(     # 수치화 되어 있는 data를 가져옴
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),    # x 데이터
    np.zeros(augment_size),     # y 데이터, 전부 0으로 채운 y값, 없으면 x만 numpy 있으면 x, y 튜플
    batch_size=augment_size,    # 통배치
    shuffle=False,
)#.next()                        # .next()는 첫번째 배치만 가져옴, 100개의 데이터에 32 배치면 x, y 32개만 가져옴

print(xy_data)                  # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000022F79620F40>
print(type(xy_data))            # <class 'keras.preprocessing.image.NumpyArrayIterator'>

print(len(xy_data))             # 1
print(xy_data[0][0].shape)      # (100, 28, 28, 1)
print(xy_data[0][1].shape)      # (100,)

plt.figure(figsize=(7,7))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(xy_data[0][0][i], cmap='gray')
plt.show()