import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    # defalut : tf.float32
node3 = tf.add(node1, node2)

print(node3)            # Tensor("Add:0", shape=(), dtype=float32)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(node3))  # 7.0

