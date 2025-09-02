import tensorflow as tf
import pandas as pd
tf.compat.v1.random.set_random_seed(518)

#1. 데이터
path = './_data/dacon/00 따릉이/'

x_csv = pd.read_csv(path + 'train.csv', index_col=0)
x_csv = x_csv.dropna()

x_data = x_csv.drop(['count'], axis=1)
y_data = x_csv['count']

print(x_data.shape, y_data.shape) # (1459, 9) (1459,)

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 9],)
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 1]),
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
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)

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
    
y_prd = x_data @ w_val + b_val

from sklearn.metrics import r2_score, mean_squared_error

print('R2S :', r2_score(y_data, y_prd))
print('MSE :', mean_squared_error(y_data, y_prd))

# R2S : 0.014161404117080112
# MSE : 6770.0880563523515