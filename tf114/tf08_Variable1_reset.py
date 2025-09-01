import tensorflow as tf
tf.random.set_random_seed(3112)

variable = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')
print(variable)

# 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(variable)
print('aaa :', aaa)
sess.close()

# 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = variable.eval(session=sess)  # tensorflow 데이터형인 'variable(변수)' 를 파이썬에서 쓸 수 있게 바꿔준다
print('bbb :', bbb)
sess.close()

# 초기화 세번쨰
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = variable.eval()
print('ccc :', ccc)
sess.close()