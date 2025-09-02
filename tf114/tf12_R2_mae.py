import tensorflow as tf

tf.random.set_random_seed(518)

import matplotlib.pyplot as plt

#1 데이터
x_data = [ 1, 2, 3, 4,  5]
y_data = [ 3, 5, 7, 9, 11]
x_test_data = [6, 7, 8]

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

#2 모델 구성
hypothesis = x * w + b

#3 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optm.minimize(loss)

loss_val_list = []
w_val_list = []

epochs = 1000*2

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data,
                                                        y:y_data})
        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    #4 평가예측
    result = x_test * w_val + b_val
    print("6 7 8 =", sess.run(result, feed_dict={x_test:x_test_data}))

### 실습 : R2, Mae 만들기 ###

from sklearn.metrics import r2_score, mean_absolute_error

y_pred = x_data * w_val + b_val

print('R2S :', r2_score(y_pred, y_data))
print('MAE :', mean_absolute_error(y_pred, y_data))

# R2S : 0.999999999999936
# MAE : 6.198883056640625e-07