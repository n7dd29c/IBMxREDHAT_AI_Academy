# keras14 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time # 시간에 대한 모듈

#1. 데이터

x_train = np.array(range(100))
y_train = np.array(range(100))

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()    # 현재 시간을 반환, 시작시간
print(start_time)           # 1747968507.3788538, 1970년 1월 1일 이후로 지난 초
model.fit(x_train, y_train, epochs = 1000, batch_size = 128, verbose=1)
# verbose = -1 : epoch 진행상황 출력
# verbose = 0 : 침묵
# verbose = 1 : default
# verbose = 2 : progress bar 삭제
# verbose = 3 : -1과 동일
end_time = time.time()
print('runtime : ', end_time - start_time, '초')

# 1000 epoch에서 verbose= 0, 1, 2, 3 의 결과값
# -------------------------------------------
# verbose=0 - runtime :  44.83443093299866 초
# verbose=1 - runtime :  56.68903136253357 초
# verbose=2 - runtime :  44.25486254692077 초
# verbose=3 - runtime :  44.28255510330200 초

# 1000 epoch에서 batch 1, 32, 64, 128 일 때 verbose=1의 결과값
# -----------------------------------------------------------
# batch_size=1 -------------- runtime :  56.68903136253357 초
# batch_size=32 ------------- runtime :  3.648700714111328 초
# batch_size=64 ------------- runtime :  2.421514749526977 초
# batch_size=128 ------------ runtime :  1.795891761779785 초
