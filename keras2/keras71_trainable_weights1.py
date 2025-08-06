import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense

SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
#  dense_1 (Dense)             (None, 2)                 8         
#  dense_2 (Dense)             (None, 1)                 3         
# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0

print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.5790334 , -0.61152977,  0.5714569 ]], dtype=float32)>,
#  tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
#
#  <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
#                                                                     array([[ 1.0207715 , -0.9624047 ],
#                                                                         [ 0.57199144,  0.55234826],
#                                                                         [-0.25543672, -0.8761578 ]], dtype=float32)>,
# 
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
# 
# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
#                                                                     array([[-0.87500095],
#                                                                         [-0.083938  ]], dtype=float32)>,
# 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

print('====================================================================================================================================')
print(model.trainable_weights)  # model.weights랑 같은 기능
print('====================================================================================================================================')

print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

####################################### 동결 #######################################
model.trainable = False                 # 동결코드
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print('동결 후')
print(model.weights)                    # 정상
print(model.trainable_weights)          # [], 나오지 않는다

# model.summary()
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17 <- 역전파 하지 않겠다 (가중치 갱신을 하지 않겠다)