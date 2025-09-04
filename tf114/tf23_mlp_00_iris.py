import tensorflow as tf
from sklearn.datasets import load_iris

tf.compat.v1.random.set_random_seed(518)

#1. 데이터
x_data, y_data = load_iris(return_X_y=True)
print(x_data.shape, y_data.shape)   # (150, 4) (150,)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.int32, shape=[None])

#2. 모델
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3]), dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), dtype=tf.float32)

Layer_1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1) 


w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 4]), dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([4]), dtype=tf.float32)

Layer_2 = tf.compat.v1.sigmoid(tf.matmul(Layer_1, w2) + b2)


#################### [[ DROPOUT ]] ####################
drop_trn = 0.2
drop_tst = 0.0
drop = tf.compat.v1.placeholder(tf.float32)

Layer_2 = tf.nn.dropout(Layer_2, rate = drop)
# 평가에서는 dropout을 적용하면 안됨
# tensor1 : dropout 비율을 상수로 지정 후 분리
# tensor2 : 자동으로 처리
# torch   : grad_zero, no_grad로 처리
#######################################################

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3]), dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), dtype=tf.float32)

hypothesis = tf.nn.softmax(Layer_2 @ w3 + b3)

#3. 컴파일
optm = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.matmul(Layer_2, w3) + b3)
loss = tf.reduce_mean(loss_vec)   # 또는 reduce_sum
train = optm.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20000
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data, drop: drop_trn})
    if step %50 == 0:
        print("epochs :", step, "loss :", cost_val)

#4. 평가 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_data, drop: drop_tst})

pred_cls = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x: x_data, drop: 0.0})
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, pred_cls)
print("acc :", acc)

# epochs : 9950 loss : 0.0005480265
# acc : 1.0