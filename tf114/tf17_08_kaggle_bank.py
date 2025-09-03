from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자 데이터 수치화
le_geo = LabelEncoder()
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])  # fit()은 train만!
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(train_csv['Gender'])     # fit()은 train만!
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv.shape)  # (165034, 11)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=111,
)

# === 추가: 표준화(스케일 정리) ===
mu = x_train.mean()
sigma = x_train.std().replace(0, 1)   # 혹시 0표준편차 방지
x_train = ((x_train - mu) / sigma)

# 넘파이 float32로 맞춰주기 (TF placeholder가 float32)
x_train = x_train.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)  # (N,)

print(x_train.shape, y_train.shape)  # (652, 8) (652,)

# 플레이스홀더 (구조 유지)
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None,])

# 파라미터 (초기 표준편차만 작게)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], stddev=0.01), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델 (구조 유지)
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# NaN 방지 및 모양 맞춤 추가 (그대로)
h = tf.squeeze(hypothesis)   # (N,1) -> (N,)
eps = 1e-7                   # log(0) 방지

# 3-1. 컴파일 (구조 유지, eps만)
loss = -tf.reduce_mean(y * tf.log(h + eps) + (1 - y) * tf.log(1 - h + eps))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=5e-3)  # 1e-2 -> 5e-3
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_train, y: y_train})
    if step % 100 == 0:
        print(step, 'loss :', cost_val)

# 예측/정확도
x_test = ((x_test - mu) / sigma).values.astype(np.float32)
y_test = y_test.values.astype(np.float32)

pred = tf.round(h)  # (N,)
print(sess.run(pred, feed_dict={x: x_test, y: y_test}))

acc = tf.cast(tf.equal(pred, y), dtype=tf.float32)
acc = tf.reduce_mean(acc)
print('acc :', sess.run(acc, feed_dict={x: x_test, y: y_test}))
