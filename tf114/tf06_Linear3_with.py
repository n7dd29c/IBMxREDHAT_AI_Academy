from xml.parsers.expat import model
import tensorflow as tf

#1. 데이터
x = [1,2,3,4,5]
y = [4,6,8,10,12]

w = tf.Variable(0.1, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델구성
# y = wx +b
hypothesis = x * w + b

#3-1. 컴파일
# model.complie(loss='mse', optimizer='adam)
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.062)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs = 1000
    for step in range(epochs):
        sess.run(train)
        if step % 100 == 0:
            print(step, '\t', sess.run(loss), '\t', sess.run(w), '\t', sess.run(b))
# with를 쓰면 sess.close()가 자동으로 실행됨