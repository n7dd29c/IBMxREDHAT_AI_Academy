import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 3)                 6
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 8
# _________________________________________________________________
# dense_2 (Dense)              (None, 4)                 12
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 31          # 전체 훈련 수
# Trainable params: 31      # 학습시키면서 업데이트된 파라미터
# Non-trainable params: 0   # 훈련엔 관여하지만 학습되지는 않는 파라미터