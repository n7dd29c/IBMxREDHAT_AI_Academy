from xml.parsers.expat import model
import tensorflow as tf
import numpy as np

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

x_test_data = [6,7,8]
# [실습] x_test_data로 14,16,18을 예측

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델구성
# y = wx +b
hypothesis = x * w + b

#3-1. 컴파일
# model.complie(loss='mse', optimizer='adam')
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.062)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # model.fit()
    epochs = 1000
    for step in range(epochs):
        _, loss_val = sess.run([train, loss],
                                             feed_dict={x:x_data, y:y_data})
        if step % 10 == 0:
            # print(step, '\t', sess.run(loss), '\t', sess.run(w), '\t', sess.run(b))
            w_val = w.eval(session=sess)
            b_val = b.eval(session=sess)
            print(step, loss_val, w_val, b_val)

    #4. 예측
    x_test = tf.placeholder(tf.float32, shape=[None])
    y_pred = hypothesis.eval(feed_dict={x:x_test}, session=sess)
    print(sess.run(y_pred, feed_dict={x_test:x_test_data}))

# with를 쓰면 sess.close()가 자동으로 실행됨

