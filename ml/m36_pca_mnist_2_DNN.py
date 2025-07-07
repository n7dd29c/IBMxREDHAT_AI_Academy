# m36_1 에서 뽑은 4가지 결과로 4개의 모델 만들기
# input_shape=
# (70000, 713)
# (70000, 331)
# (70000, 486)
# (70000, 154)

# 힌트 : 
n_components_list = [154, 331, 486, 713, 784]

import time
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(x_train.shape, x_test.shape)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

all_results = []

for n_components in n_components_list:
    print(f"\n--- PCA n_components: {n_components} 로 모델 훈련 시작 ---")

    # PCA 적용을 위해 원본 데이터를 다시 로드 (이전 루프에서 데이터가 변형되므로)
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()
    x_train_orig = x_train_orig / 255.
    x_test_orig = x_test_orig / 255.
    x_train_orig = x_train_orig.reshape(x_train_orig.shape[0], 28 * 28)
    x_test_orig = x_test_orig.reshape(x_test_orig.shape[0], x_test_orig.shape[1] * x_test_orig.shape[2])

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_orig)
    x_test_pca = pca.transform(x_test_orig)

    print(f"PCA 적용 후 x_train_pca.shape: {x_train_pca.shape}, x_test_pca.shape: {x_test_pca.shape}")

    model = Sequential([
        Dense(1024, input_dim=n_components, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True,
    )
    start_time = time.time()
    model.fit(x_train_pca, y_train, epochs=300, batch_size=64, callbacks=es, validation_split=0.2, verbose=2)
    end_time = time.time() - start_time

    #4. 평가
    results = model.evaluate(x_test_pca, y_test, verbose=0)
    
    model_result = {
            'n_components': n_components,
            'loss': results[0],
            'accuracy': results[1],
            'time_taken': np.round(end_time, 2)
        }
    all_results.append(model_result)
        
    print(f'현재 모델 (n_components={n_components}) 훈련 및 평가 완료. 소요 시간: {np.round(end_time, 2)} 초')

for result in all_results:
    print(f"--- n_components: {result['n_components']} ---")
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Time Taken: {result['time_taken']} 초")
    print("----------------------------")
    
# --- n_components: 154 ---
#   Loss: 0.0812
#   Accuracy: 0.9779
#   Time Taken: 52.64 초
# ----------------------------
# --- n_components: 331 ---
#   Loss: 0.0881
#   Accuracy: 0.9736
#   Time Taken: 42.48 초
# ----------------------------
# --- n_components: 486 ---
#   Loss: 0.0845
#   Accuracy: 0.9756
#   Time Taken: 51.57 초
# ----------------------------
# --- n_components: 713 ---
#   Loss: 0.0884
#   Accuracy: 0.9722
#   Time Taken: 56.4 초
# ----------------------------
# --- n_components: 784 ---
#   Loss: 0.0926
#   Accuracy: 0.9739
#   Time Taken: 58.52 초
# ----------------------------