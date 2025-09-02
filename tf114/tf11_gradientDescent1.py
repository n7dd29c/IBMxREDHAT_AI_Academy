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
# optm = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optm.minimize(loss)

# loss = (hypothesis - y)**2 = (x*w-y)**2
# 미분loss = 2 * (x * w - y) * x
lr = 0.0005
gradient = tf.reduce_mean((x*w - y)*x)  # w = w - lr * (dL/dW) 중에서 "dL/dW"
descent = w - lr * gradient
train = w.assign(descent)       # assign : 기존 값을 대체한다

loss_val_list = []
w_val_list = []

epochs = 1000

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
    y_pred = x_test * w_val + b_val
    print("6 7 8 =", sess.run(y_pred, feed_dict={x_test:x_test_data}))