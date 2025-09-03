import tensorflow as tf
import numpy as np
tf.random.set_random_seed(3112)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias', dtype=tf.float32)

#2. 모델
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)

train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

pred_class = tf.argmax(hypothesis, axis=1)   # shape: (N,)

# 정답 클래스: one-hot y에서 argmax
true_class = tf.argmax(y, axis=1)            # shape: (N,)

# 정확도
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32))

#3-2. 훈련
# 다중분류용 예측/정확도 (argmax)
pred_class  = tf.argmax(hypothesis, axis=1)  # (N,)
true_class  = tf.argmax(y, axis=1)           # (N,)
accuracy    = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32))

# 3-2. 훈련
epochs = 10001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 500 == 0:
        acc_val = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
        print(step, "loss:", cost_val, "acc:", acc_val)

# 최종 예측/정확도
pred_cls_val, acc_val = sess.run([pred_class, accuracy], feed_dict={x: x_data, y: y_data})
print("pred class:", pred_cls_val)
print("acc:", acc_val)

# (선택) sklearn로도 확인하고 싶다면:
from sklearn.metrics import accuracy_score
print("acc(sklearn):", accuracy_score(np.argmax(y_data, axis=1), pred_cls_val))