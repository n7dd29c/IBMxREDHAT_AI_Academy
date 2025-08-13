# encoder
# Input   (28, 28,  1)
# Conv    (28, 28, 64) padding='same'
# MaxPool (14, 14, 64)
# Conv    (14, 14, 32) padding='same'   # ← 일관성 위해 'valid'를 'same'으로 조정
# MaxPool ( 7,  7, 32)

# decoder
# Conv         ( 7,  7, 32) padding='same'
# UpSampling2D (14, 14, 32)
# Conv         (14, 14, 16) padding='same'
# UpSampling2D (28, 28, 16)
# Conv         (28, 28,  1) padding='same', activation='sigmoid'

import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt

# 1) 데이터: [0,1] 정규화 + 채널 차원 추가
(x_train_raw, _), (x_test_raw, _) = mnist.load_data()
x_train = (x_train_raw.astype('float32') / 255.0)[..., np.newaxis]  # (60000, 28, 28, 1)
x_test  = (x_test_raw.astype('float32')  / 255.0)[..., np.newaxis]  # (10000, 28, 28, 1)

# 노이즈 추가 (가우시안) + 클리핑
rng = np.random.default_rng(42)
x_train_noised = np.clip(x_train + rng.normal(0, 0.1, size=x_train.shape), 0, 1)
x_test_noised  = np.clip(x_test  + rng.normal(0, 0.1, size=x_test.shape), 0, 1)

# 2) 모델
inp = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(64, (3,3), padding='same', activation='relu')(inp)   # (28,28,64)
x = MaxPooling2D(pool_size=(2,2))(x)                            # (14,14,64)
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)     # (14,14,32)
x = MaxPooling2D(pool_size=(2,2))(x)                            # (7,7,32)

# Decoder
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)     # (7,7,32)
x = UpSampling2D(size=(2,2))(x)                                 # (14,14,32)
x = Conv2D(16, (3,3), padding='same', activation='relu')(x)     # (14,14,16)
x = UpSampling2D(size=(2,2))(x)                                 # (28,28,16)
out = Conv2D(1, (3,3), padding='same', activation='sigmoid')(x) # (28,28,1)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 3) 훈련 (입력=노이즈, 타깃=클린)
autoencoder.fit(
    x_train_noised, x_train,
    epochs=20, batch_size=128, validation_split=0.2, verbose=1
)

# 4) 예측
decoded = autoencoder.predict(x_test_noised, verbose=0)

# 5) 시각화
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].squeeze(), cmap='gray')
    ax.axis('off')
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
