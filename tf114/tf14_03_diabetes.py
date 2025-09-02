import tensorflow as tf
from sklearn.datasets import load_diabetes
tf.compat.v1.random.set_random_seed(518)

#1. 데이터
x_trn, y_trn = load_diabetes(return_X_y=True)
print(x_trn.shape, y_trn.shape) # (442, 10) (442,)


x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10],)
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]),
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
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.5)

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
    
y_prd = x_trn@ w_val + b_val

from sklearn.metrics import r2_score, mean_squared_error

print('R2S :', r2_score(y_trn, y_prd))
print('MSE :', mean_squared_error(y_trn, y_prd))

# R2S : -9.949689383503824e-07
# MSE : 5929.890796961663