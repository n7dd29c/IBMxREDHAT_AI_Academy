# keras47 copy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array   # 가져온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np

path = './_data/image/me/'
img = load_img(path + 'me.jpg', target_size=(100,100), )
print(img)          # <PIL.Image.Image image mode=RGB size=100x100 at 0x22AD9C451C0>
print(type(img))    # <class 'PIL.Image.Image'>  PIL : Python Image Library의 약자

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>

######### 3차원 -> 4차원 변환 #########
arr = arr.reshape(1, 100, 100, 3)
print(arr)
print(arr.shape)    # (1, 150, 100, 3)

# img = np.expand_dims(arr, axis=0) # axis = 1을 넣을 번지수
# print(img.shape)    # (1, 150, 150, 3)

# np.save(path + 'keras47_me.npy', arr=img)

############ 여기부터 증폭 ############

datagen = ImageDataGenerator(
    rescale=1./255,         # 0~255 정규화 (scaling)
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,  # 수치 평행이동 10%
    height_shift_range=0.1,
    zoom_range=1.2,
    rotation_range=15,      # 5도 회전
    fill_mode='nearest',    # 빈 공간 근접치로 채움
)

it = datagen.flow(
    arr,
    batch_size=1,
)

print('============================================================================')
print(it)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001CC9EAE69A0>
print('============================================================================')
# aaa = it.next()     # 파이썬 2.0 문법, 잘 쓰지 않음
# print(aaa)
# print(aaa.shape)    # (1, 100, 100, 3)

bbb = next(it)
print(bbb)
print(bbb.shape)    # (1, 100, 100, 3)

# print(it.next())
# print(it.next())
# print(it.next())

# fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5,5))
# for row in range(2):
#     # batch = it.next() # IDG에서 랜덤으로 한번 작업 (변환)
#     for col in range(5):
#         batch = next(it)
#         print(batch.shape)  # (1, 100, 100, 3)
#         batch = batch.reshape(100, 100, 3)
#         ax[row, col].imshow(batch)
#         ax[row, col].axis('off')
# plt.show()

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,5))
ax = ax.flatten()       # 2차원 변환
for i in range(10):
    batch = next(it)
    batch = batch.reshape(100, 100, 3)
    ax[i].imshow(batch)
    ax[i].axis('off')
plt.tight_layout()
plt.show()