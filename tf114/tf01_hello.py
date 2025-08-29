import tensorflow as tf
print(tf.__version__)   # 1.14.0
# python : 3.7.16

### tensorflow 설치오류 시 - 20250829 기준
# pip install protobuf==3.20
# pip install numpy==1.16

# 일반적인 경우
print('hello world')

# tensorflow-1 의 경우
hello = tf.constant('hello world')
print(hello)            # Tensor("Const:0", shape=(), dtype=string), 원하는 결과가 아님

sess = tf.Session()
print(sess.run(hello))  # 그래프 연산을 실행시킴
# b'hello world' <- b는 binary, 원하는 결과

# python = input -> output
# tensorflow = input -> tensorflow session -> output