import tensorflow as tf

print('tf version :', tf.__version__)           # tf version : 1.14.0
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : False

# 가상환경 tf114cpu -> cpu_venv로 변경
# tf version : 2.12.0
# 즉시실행모드 : True
# 1.x.x 에서 2.x.x로 넘어오면서 true?

tf.compat.v1.disable_eager_execution()
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : False

tf.compat.v1.enable_eager_execution()
print('즉시실행모드 :', tf.executing_eagerly()) # 즉시실행모드 : True

hello = tf.constant('hello world')
sess = tf.compat.v1.Session()
# print(sess.run(hello))    # 오류
print(hello)                # tf.Tensor(b'hello world', shape=(), dtype=string)

#############################################################################################

# 즉시실행모드 -> tensorflow-1 그래프 형태의 구성 없이 자연스러운 python 문법으로 실행
# tf.compat.v1.disable_eager_execution()  # 즉시실행모드 끄기 // tensorflow 1.0 문법(defalut)
# tf.compat.v1.enable_eager_execution()   # 즉시실행모드 켜기 // tensorflow 2.0 부터 사용가능

# sess.run() 실행 시
# 가상환경            즉시실행모드            사용가능
# 1.14.0                disable            b'hello world'
# 1.14.0                enable                 error
# 2.12.0                disable            b'hello world'
# 2.12.0                enable                 error

# Tensorflow-1 은 '그래프 연산' 모드
# Tensorflow-2 는 '즉시실행' 모드

# tf.compat.v1.enable_eager_execution()   # 즉시실행모드 활성화
# -> Tensorflow-2 의 기본값

# tf.compat.v1.disable_eager_execution()  # 즉시실행모드 비활성화
# -> '그래프 연산' 모드로 돌아감
# -> Tensorflow-1 코드를 쓸 수 있음

# tf.executing_eagerly()
# -> True : 즉시실행모드, Tensorflow-2 코드만 써야함
# -> False : 그래프 연산 모드, Tensorflow-1 코드를 쓸 수 있음

#############################################################################################
