import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.decomposition import PCA

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
                                 # (평균, 표준편차 0.1인 정규분포형태의 랜덤값, size)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)        # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.4611409472854433 -0.5246125806788717

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.0 0.0

#2. 모델
input_img = Input(shape=(28*28,))

def autoencoder(hidden_layer_size):
    model = Sequential([
        Dense(units=hidden_layer_size, input_shape=(28*28,)),
        Dense(784, activation='sigmoid')
    ])
    return model

hidden_size = [1,8,32,64,154,331,486,713]
outputs = []
outputs.append(x_test)

for i in hidden_size:
    #3. 컴파일, 훈련
    print(f'======================================= model_{i} =======================================')
    model = autoencoder(hidden_layer_size=i)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)
    
    #4. 평가, 예측
    decoded_img = model.predict(x_test_noised)
    outputs.append(decoded_img)

import matplotlib.pyplot as plt
import random
fig, axes = plt.subplots(9, 5, figsize=(15,8))

random_images = random.sample(range(decoded_img.shape[0]), 5)

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()