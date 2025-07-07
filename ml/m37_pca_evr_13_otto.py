from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

path = './Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# print(train_csv.shape)  # (200000, 201)
# print(test_csv.shape)   # (200000, 200)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333,
    # stratify=y
)

print(x_train.shape)    # (16512, 8)
print(x_test.shape)     # (4128, 8)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=x_train_scaled.shape[1])
pca = pca.fit(x_train_scaled)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

x1 = np.argmax(evr_cumsum>=1.0)+1
x2 = np.argmax(evr_cumsum>=0.999)+1
x3 = np.argmax(evr_cumsum>=0.99)+1
x4 = np.argmax(evr_cumsum>=0.95)+1

print('1 이상 : ', x1)      # 1
print('0.999이상 : ', x2)   # 93
print('0.99이상 : ', x3)    # 89
print('0.95이상 : ', x4)    # 77
 
n_components_list = [x1, x2, x3, x4]
print(n_components_list)
# exit()
# =======================================================================

import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

all_results = []

for n_components in n_components_list:
    print(f"\n--- PCA n_components: {n_components} 로 모델 훈련 시작 ---")

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    print(f"PCA 적용 후 x_train_pca.shape: {x_train_pca.shape}, x_test_pca.shape: {x_test_pca.shape}")

    model = Sequential([
        Dense(1024, input_dim=n_components, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(1)
    ])

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
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
    
# --- n_components: 8 ---
#   Loss: 0.2581
#   Accuracy: 0.2581
#   Time Taken: 103.17 초
# ----------------------------
# --- n_components: 8 ---
#   Loss: 0.2544
#   Accuracy: 0.2544
#   Time Taken: 77.78 초
# ----------------------------
# --- n_components: 7 ---
#   Loss: 0.3095
#   Accuracy: 0.3095
#   Time Taken: 53.4 초
# ----------------------------
# --- n_components: 6 ---
#   Loss: 0.3660
#   Accuracy: 0.3660
#   Time Taken: 66.71 초
# ----------------------------