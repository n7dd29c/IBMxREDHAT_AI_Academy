import tensorflow as tf
from sklearn.datasets import fetch_california_housing
tf.compat.v1.random.set_random_seed(518)

#1. 데이터
DS = fetch_california_housing()
x_data = DS.data
y_data = DS.target
print(x_data.shape, y_data.shape) # (20640, 8) (20640,)

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1]),
                          dtype=tf.float32,
                          name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),
                          dtype=tf.float32,
                          name='bias')

#2.모델
hypothesis = tf.compat.v1.matmul(x, w) + b
hypothesis = x @ w + b

#3 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

train= optm.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    epochs = 1000
    
    for step in range(epochs):
        _, loss_val, w_val, b_val = \
            sess.run([train, loss, w, b],
                              feed_dict = {x:x_data,
                                           y:y_data})
        
        if step % 100 == 0 :
            print(step, loss_val)
    
y_prd = x_data @ w_val + b_val

from sklearn.metrics import r2_score, mean_squared_error

print('R2S :', r2_score(y_data, y_prd))
print('MSE :', mean_squared_error(y_data, y_prd))

# R2S : -0.06727513601050106
# MSE : 1.4211305276246349