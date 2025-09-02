import tensorflow as tf

tf.random.set_random_seed(518)

import matplotlib.pyplot as plt

#1 데이터
x_data = [ 1, 2, 3, 4,  5]
y_data = [ 3, 5, 7, 9, 11]
x_test_data = [6, 7, 8]

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

#2 모델 구성
hypothesis = x * w + b

#3 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optm = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optm.minimize(loss)

loss_val_list = []
w_val_list = []

epochs = 1000

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data,
                                                        y:y_data})
        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    #4 평가예측
    y_pred = x_test * w_val + b_val
    print("6 7 8 =", sess.run(y_pred, feed_dict={x_test:x_test_data}))

print("=================그래프=================")
# print(loss_val_list)
# print(w_val_list)

# loss - epochs 그래프
# plt.plot(loss_val_list)
# plt.show()

# weight - epochs 그래프
# plt.plot(w_val_list)
# plt.xlabel("epoch")
# plt.ylabel("weights")
# plt.grid()
# plt.show()

# weight - loss 그래프
# plt.plot(w_val_list, loss_val_list)
# plt.xlabel("weights")
# plt.ylabel("loss")
# plt.grid()
# plt.show()

# [실습] Subplot으로 위 3개의 그래프 한번에 출력
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

# 1. loss - epochs
ax[0].plot(loss_val_list)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
ax[0].set_title("Loss vs Epochs")
ax[0].grid()

# 2. weight - epochs
ax[1].plot(w_val_list)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("weights")
ax[1].set_title("Weights vs Epochs")
ax[1].grid()

# 3. weight - loss
ax[2].plot(w_val_list, loss_val_list)
ax[2].set_xlabel("weights")
ax[2].set_ylabel("loss")
ax[2].set_title("Loss vs Weights")
ax[2].grid()

plt.tight_layout()
plt.show()