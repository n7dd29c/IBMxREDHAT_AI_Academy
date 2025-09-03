import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_covtype

tf.compat.v1.set_random_seed(3112)

# 1. 데이터
x_data, y_data = fetch_covtype(return_X_y=True)   # (581012, 54) (581012,)

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.int32)

# === [추가] 간단 표준화 ===
mu = x_data.mean(axis=0, keepdims=True)
sigma = x_data.std(axis=0, keepdims=True)
sigma[sigma == 0] = 1.0
x_data = (x_data - mu) / sigma

# placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.int32,   shape=[None])     # 정수 레이블(0,1,2)

# 2. 파라미터
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([54, 3], stddev=0.01), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias')

# 3. 모델
logits = tf.matmul(x, w) + b               # (N,3)
hypothesis = tf.nn.softmax(logits)         # (N,3)

# 4. 손실 (one-hot + eps)
y_one = tf.one_hot(y, depth=3)             # (N,3)
eps = 1e-7
loss = tf.reduce_mean(-tf.reduce_sum(y_one * tf.math.log(hypothesis + eps), axis=1))

# [수정] 학습률 더 낮춤
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

# 5. 예측/정확도
pred_class = tf.argmax(hypothesis, axis=1, output_type=tf.int32)
accuracy   = tf.reduce_mean(tf.cast(tf.equal(pred_class, y), tf.float32))

# 6. 학습
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if step % 500 == 0:
        acc_val = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
        print(f"{step:4d} | loss: {loss_val:.4f} | acc: {acc_val:.4f}")

# 최종 평가
pred_cls_val, acc_val = sess.run([pred_class, accuracy], feed_dict={x: x_data, y: y_data})
print("pred class:", pred_cls_val)
print("acc:", acc_val)
