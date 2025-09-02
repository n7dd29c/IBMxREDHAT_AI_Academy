import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
tf.compat.v1.random.set_random_seed(518)

#1. 데이터
(x_trn, y_trn), (x_tst, y_tst) = boston_housing.load_data()
print(x_trn.shape, y_trn.shape) # (404, 13) (404,)
print(x_tst.shape, y_tst.shape)

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 13],)
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1]),
                          dtype=tf.float32,
                          name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),
                          dtype=tf.float32,
                          name='bias')

#2.모델
# hypothesis = tf.compat.v1.matmul(x, w) + b
hypothesis = x @ w + b

#3 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

train= optm.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    epochs = 5000
    
    for step in range(epochs):
        _, loss_val, w_val, b_val = \
            sess.run([train, loss, w, b],
                              feed_dict = {x:x_trn,
                                           y:y_trn})
        
        if step % 100 == 0 :
            print(step, loss_val)

# 행렬 연산 : Tensor            
y_prd = tf.compat.v1.matmul(tf.cast(x_tst, tf.float32), w_val) + b_val

# 행렬 연산 : Numpy
y_prd = np.matmul(x_tst, w_val) + b_val

# 행렬 연산 : Python
y_prd = x_tst @ w_val + b_val
### Python 이나 Tensor 형태 연산은 그래프 포함

from sklearn.metrics import r2_score, mean_squared_error

print('R2S :', r2_score(y_tst, y_prd))
print('MSE :', mean_squared_error(y_tst, y_prd))