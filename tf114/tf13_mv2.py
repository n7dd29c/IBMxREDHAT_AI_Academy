import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_random_seed(518)

#1. 데이터
x1_data = [ 73.,  93.,  89.,  96.,  73.]
x2_data = [ 80.,  88.,  91.,  98.,  66.]
x3_data = [ 75.,  93.,  90., 100.,  70.]

x_data = [[73.,  93.,  89.,  96.,  73.],
          [80.,  88.,  91.,  98.,  66.],
          [75.,  93.,  90., 100.,  70.]]
x_data = np.array(x_data).T

y_data = [152., 185., 180., 196., 142.]
y_data = np.array(y_data).reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.compat.v1.random_normal([3, 1]), dtype=tf.float32)
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name="bias")

#2. 모델
# hypothesis = x @ w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

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
                              feed_dict = {x:x_data,
                                           y:y_data})
        
        if step % 100 == 0 :
            print(step, loss_val)
    
y_pred = x_data @ w_val + b_val

from sklearn.metrics import r2_score, mean_absolute_error

print('R2S :', r2_score(y_data, y_pred))
print('MAE :', mean_absolute_error(y_data, y_pred))

# R2S : 0.9995323025543879
# MAE : 0.3798646092414856
