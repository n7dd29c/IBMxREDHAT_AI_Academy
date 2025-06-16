from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array   # 가져온 이미지 수치화
import matplotlib.pyplot as plt
import numpy as np

path = './_data/image/me/'
img = load_img(path + 'haerin.jpeg', target_size=(100,100), )
print(img)          # <PIL.Image.Image image mode=RGB size=100x100 at 0x22AD9C451C0>
print(type(img))    # <class 'PIL.Image.Image'>  PIL : Python Image Library의 약자

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>

######### 3차원 -> 4차원 변환 #########
# arr = arr.reshape(1, 100, 100, 3)
# print(arr)
# print(arr.shape)    # (1, 100, 100, 3)

img = np.expand_dims(arr, axis=0) # axis = 1을 넣을 번지수
print(img.shape)    # (1, 100, 100, 3)

np.save(path + 'keras47_haerin_2.npy', arr=img)