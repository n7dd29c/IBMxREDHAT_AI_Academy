import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.compat.v1.random.set_random_seed(518)

#1. 데이터
path = './_data/dacon/따릉이/'

x_csv = pd.read_csv(path + 'train.csv', index_col=0)
x_csv = x_csv.dropna()

x_data = x_csv.drop(['count'], axis=1)
y_data = x_csv['count']

print(x_data.shape, y_data.shape)   # (1328, 9) (1328,)

# ★ (1) 수동 표준화: 넘파이만 사용
x_mu, x_sigma = x_data.mean(axis=0), x_data.std(axis=0) + 1e-8
y_mu, y_sigma = y_data.mean(),       y_data.std()       + 1e-8

x_scaled = ((x_data - x_mu) / x_sigma).astype(np.float32)
y_scaled = ((y_data - y_mu) / y_sigma).to_numpy().reshape(-1, 1).astype(np.float32)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

#2. 모델
# ★ (2) 초기 가중치의 표준편차를 작게
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 16], stddev=0.05), dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), dtype=tf.float32)
Layer_1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1) 

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 8], stddev=0.05), dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), dtype=tf.float32)
Layer_2 = tf.nn.relu(tf.matmul(Layer_1, w2) + b2)

#################### [[ DROPOUT ]] ####################
drop_trn = 0.05   # ★ 회귀는 우선 0.0으로 학습 안정화 후 필요 시 아주 작게(0.05) 시도
drop_tst = 0.0
drop = tf.compat.v1.placeholder(tf.float32)
Layer_2 = tf.nn.dropout(Layer_2, rate=drop)
#######################################################

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1], stddev=0.05), dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)
hypothesis = tf.compat.v1.matmul(Layer_2, w3) + b3

#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optm.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    cost_val, _ = sess.run(
        [loss, train],
        feed_dict={x: x_scaled, y: y_scaled, drop: drop_trn}   # ★ 스케일된 값으로 학습
    )
    if step % 200 == 0:
        print("epoch:", step, "loss:", cost_val)

#4. 평가 예측 (원 스케일로 복원)
y_pred_scaled = sess.run(hypothesis, feed_dict={x: x_scaled, drop: drop_tst}).squeeze()
y_pred = (y_pred_scaled * y_sigma) + y_mu    # ★ 복원

mse = mean_squared_error(y_data, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_data, y_pred)
r2  = r2_score(y_data, y_pred)
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
