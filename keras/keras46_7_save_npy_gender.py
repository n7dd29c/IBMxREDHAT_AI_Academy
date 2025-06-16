import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = './_data/kaggle/men_women/'

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=True,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.7,
    zoom_range=1.2,
)

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_data_gen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    seed=333,
)

xy_test = test_data_gen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    seed=333,
)

all_x_train = []
all_y_train = []
all_x_sub = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x_train.append(x_batch)
    all_y_train.append(y_batch)
    
for i in range(len(xy_test)):
    x_batch, y_batch = xy_test[i]
    all_x_sub.append(x_batch)

x_tr = np.concatenate(all_x_train, axis=0)
y_tr = np.concatenate(all_y_train, axis=0)
x_sub = np.concatenate(all_x_sub, axis=0)

np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + 'keras46_01_x_train_gender.npy', arr=x_tr)
np.save(np_path + 'keras46_01_y_train_gender.npy', arr=y_tr)
np.save(np_path + 'keras46_01_x_sub_gender.npy', arr=x_sub)