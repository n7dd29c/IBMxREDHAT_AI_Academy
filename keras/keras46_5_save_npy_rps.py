import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = './_data/kaggle/rps'

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
)

test_data_gen = ImageDataGenerator(
    rescale=1./255,
)

xy_train = train_data_gen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb',
    seed=333,
)

xy_test = test_data_gen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb',
    seed=333,
)

print(xy_train[0][0].shape) # (64, 100, 100, 3)
print(xy_train[0][1].shape) # (64, 3)
print(len(xy_train))        # 33

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
np.save(np_path + 'keras46_01_x_train_rps.npy', arr=x_tr)
np.save(np_path + 'keras46_01_y_train_rps.npy', arr=y_tr)
np.save(np_path + 'keras46_01_x_sub_rps.npy', arr=x_sub)