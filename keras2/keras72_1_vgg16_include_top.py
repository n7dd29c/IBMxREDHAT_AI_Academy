import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, \
    ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2


SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# model = VGG16()
# model = VGG16(
#         weights='imagenet',
#         include_top=True,           # defalut : True
#         input_shape=(224, 224, 3),
# )

# model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
#  flatten (Flatten)           (None, 25088)             0         
#  fc1 (Dense)                 (None, 4096)              102764544 
#  fc2 (Dense)                 (None, 4096)              16781312  
#  predictions (Dense)         (None, 1000)              4097000   
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

# model = VGG16(
#         weights='imagenet',
#         include_top=False,           # False지정하면 fully-connected 다 날림
#         input_shape=(224,224,3),
# )
# model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
# =================================================================
# Total params: 14714688 (56.13 MB)
# Trainable params: 14714688 (56.13 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

model = VGG16(
        weights='imagenet',
        include_top=False,           # defalut : True
        input_shape=(100, 100, 3),
)

model.summary()

# =================================================================
# Total params: 14714688 (56.13 MB)
# Trainable params: 14714688 (56.13 MB)
# Non-trainable params: 0 (0.00 Byte)

# ======================= include_top=False =======================
# 1. input_shape를 우리가 훈련시킬 데이터의 shape로 수정
# 2. FC layer 없어진다 (직접 아래에 FC layer 붙여주면 된다)

