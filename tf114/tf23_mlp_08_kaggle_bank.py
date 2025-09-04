import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

tf.compat.v1.random.set_random_seed(518)

#1. 데이터
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

x_data = train_csv.drop(['Exited'], axis=1)
print(x_data.shape)  # (165034, 10)
y_data = train_csv['Exited']
print(y_data.shape)  # (165034,)

# X만 표준화 (y는 0/1 그대로)
x_mu, x_sigma = x_data.mean(axis=0), x_data.std(axis=0) + 1e-8
x_scaled = ((x_data - x_mu) / x_sigma).astype(np.float32)

# y는 float32로 (0/1), (N,1)로 reshape
y_bin = y_data.astype(np.float32).to_numpy().reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 2) 모델 (구조 그대로, 마지막만 시그모이드)
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 16], stddev=0.05), dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), dtype=tf.float32)
Layer_1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 8], stddev=0.05), dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), dtype=tf.float32)
Layer_2 = tf.nn.relu(tf.matmul(Layer_1, w2) + b2)

# 드롭아웃 (학습 때만 약하게; 필요 없으면 0.0)
drop_trn = 0.05
drop_tst = 0.0
drop = tf.compat.v1.placeholder(tf.float32)
Layer_2 = tf.nn.dropout(Layer_2, rate=drop)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1], stddev=0.05), dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

logits = tf.matmul(Layer_2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)  # (None,1)

# 3) 손실/옵티마이저 (이진 크로스엔트로피)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optm.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    cost_val, _ = sess.run(
        [loss, train],
        feed_dict={x: x_scaled, y: y_bin, drop: drop_trn}
    )
    if step % 200 == 0:
        print(f"epoch: {step:4d} | loss: {cost_val:.5f}")

# 4) 평가 (확률→0/1, accuracy)
pred_prob = sess.run(hypothesis, feed_dict={x: x_scaled, drop: drop_tst}).squeeze()
pred_cls  = (pred_prob > 0.5).astype(np.int32)
acc = accuracy_score(y_data, pred_cls)
print("accuracy:", acc)
