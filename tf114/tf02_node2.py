import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.compat.v1.Session()

# 실습
# 덧셈  add
# node_add = tf.add(node1, node2)
node_add = node1 + node2
print(sess.run(node_add))   # 5.0

# 뺄셈  subtract
# node_sub = tf.subtract(node1, node2)
node_sub = node1 - node2
print(sess.run(node_sub))   # -1.0

# 곱셈  multifly
# node_mul = tf.multiply(node1, node2)
node_mul = node1 * node2
print(sess.run(node_mul))   # 6.0

# 나눗셈  divide
# node_div = tf.divide(node1, node2)
node_div = node1 / node2
print(sess.run(node_div))   # 0.6666667



