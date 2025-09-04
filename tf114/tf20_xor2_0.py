import tensorflow as tf

tf.compat.v1.random.set_random_seed(518)

#1. 데이터
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1],]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 3]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

#2. 모델
h1 = tf.compat.v1.sigmoid(x @ w1 + b1)
hypothesis = tf.compat.v1.sigmoid(h1 @ w2 + b2)

#3. 컴파일
loss = -tf.compat.v1.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1 - hypothesis))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits=x @ w + b)
optm = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train= optm.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10000
for step in range(epochs):
    cost_val, _, w1_val, b1_val, w2_val, b2_val = \
                sess.run([loss, train, w1, b1, w2, b2],
                        feed_dict={x:x_data,
                                y:y_data})
    
    if step %50 == 0:
        print("epochs :", step, "loss :", cost_val)

#4. 평가 예측
hidd_1 = tf.sigmoid(x_data @ w1_val + b1_val)
y_pred = tf.sigmoid(hidd_1 @ w2_val + b2_val)
y_pred = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_data)
print("acc :", acc)

# epochs : 9950 loss : 0.01344231
# acc : 1.0