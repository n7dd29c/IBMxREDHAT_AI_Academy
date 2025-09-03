from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
tf.compat.v1.random.set_random_seed(3112)

# 1. 데이터
x_train, y_train = load_breast_cancer(return_X_y=True)
print(x_train.shape, y_train.shape)  # (569, 30) (569,)

# 플레이스홀더 (구조 유지)
x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None,])

# 파라미터 (구조 유지)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델 (구조 유지)
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# NaN 방지 및 모양 맞춤 추가
h = tf.squeeze(hypothesis)   # (N,1) -> (N,)
eps = 1e-7                   # log(0) 방지용 아주 작은 값

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(h + eps) + (1 - y) * tf.log(1 - h + eps))  # binary_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2) 
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val, _pred_np = sess.run(
        [loss, train, w, b, hypothesis],
        feed_dict={x: x_train, y: y_train}
    )
    if step % 20 == 0:
        print(step, 'loss :', cost_val)

print('weights :', w_val, 'bias :', b_val)

# 예측/정확도 (그래프 내에서 계산)
pred = tf.round(h)  # (N,)
print(sess.run(pred, feed_dict={x: x_train, y: y_train}))

acc = tf.cast(tf.equal(pred, y), dtype=tf.float32)
acc = tf.reduce_mean(acc)
print(sess.run(acc, feed_dict={x: x_train, y: y_train}))
