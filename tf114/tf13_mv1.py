import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_random_seed(518)

#1. 데이터
x1_data = [ 73.,  93.,  89.,  96.,  73.]
x2_data = [ 80.,  88.,  91.,  98.,  66.]
x3_data = [ 75.,  93.,  90., 100.,  70.]
y_data  = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w2 = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w3 = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32, name="bias")

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 +b

#3 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

train= optm.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    epochs = 5000
    
    for step in range(epochs):
        _, loss_val, w1_val, w2_val, w3_val, b_val = \
            sess.run([train, loss, w1, w2, w3, b],
                              feed_dict = {x1:x1_data,
                                           x2:x2_data,
                                           x3:x3_data,
                                           y:y_data})
        
        if step % 100 == 0 :
            print(step, loss_val)
    
y_pred = x1_data*w1_val + x2_data*w2_val + x3_data*w3_val + b_val

from sklearn.metrics import r2_score, mean_absolute_error

print('R2S :', r2_score(y_data, y_pred))
print('MAE :', mean_absolute_error(y_data, y_pred))
