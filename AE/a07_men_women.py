# keras46, 47 참고하여, 남자 여자 사진에 노이즈를 주고, 내 사진에도 노이즈를 추가
# autoencoder로 피부 미백 훈련 가중치를 만든다
# 그 가중치로 내 사진을 예측해서 피부 미백

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from sklearn.model_selection import train_test_split


np_path = './_data/_save_npy/'
x = np.load(np_path + 'keras46_01_x_train_gender.npy')
y = np.load(np_path + 'keras46_01_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=55, test_size=0.2, 
)
print(x_train.shape, x_test.shape)

rng = np.random.default_rng(42)
x_train_noised = np.clip(x_train + rng.normal(0, 0.1, size=x_train.shape), 0, 1)
x_test_noised  = np.clip(x_test  + rng.normal(0, 0.1, size=x_test.shape), 0, 1)
print(x_train_noised.shape, x_test_noised.shape)                # (2647, 100, 100, 3) (662, 100, 100, 3)

# 2) 모델
inp = Input(shape=(100, 100, 3))

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
out = Conv2D(3, (3,3), padding='same', activation='sigmoid')(x) # (28,28,1)

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
plt.figure(figsize=(15, 10))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].squeeze(), cmap='gray')
    ax.axis('off')
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
