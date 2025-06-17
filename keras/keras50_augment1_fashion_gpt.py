from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # 패션 MNIST 데이터 불러오기

x_train = x_train/255.   # 0~1로 정규화
x_test = x_test/255.     # 0~1로 정규화

# 데이터 증강 설정: 여러 방법으로 이미지 변형
datagen = ImageDataGenerator(
    horizontal_flip=True,    # 좌우 반전
    vertical_flip=True,      # 상하 반전
    width_shift_range=0.1,   # 가로로 10% 이동
    height_shift_range=0.1,  # 세로로 10% 이동
    zoom_range=1.2,          # 최대 20% 확대
    rotation_range=10,       # 최대 10도 회전
    fill_mode='nearest',     # 빈 공간은 주변 색으로 채움
)

augment_size = 40000  # 증강할 데이터 개수

# 원본 데이터에서 랜덤으로 40000개 샘플 선택
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)                        # 선택된 인덱스 출력
print(np.min(randidx), np.max(randidx))  # 최소, 최대 인덱스 출력

# 선택된 샘플 복사해서 새 배열 만들기 (원본 손상 방지)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape)  # (40000, 28, 28)
print(y_augmented.shape)  # (40000,)

# CNN에 맞게 차원 추가 (채널=1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],  # 40000
    x_augmented.shape[1],  # 28
    x_augmented.shape[2],  # 28
    1,                     # 채널
)
print(x_augmented.shape)   # (40000, 28, 28, 1)

# 설정한 datagen으로 증강 데이터 생성 (이미지 배치만 사용)
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,           # 라벨도 같이 주지만 아래에서 [0]로 이미지만 받음
    batch_size=augment_size,
    shuffle=False,
).next()[0]                # 이미지 배치만 가져오기

print(x_augmented.shape)   # (40000, 28, 28, 1)

# 원본 훈련 데이터 모양도 CNN에 맞게 reshape
print(x_train.shape)       # (60000, 28, 28)
x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 원본 + 증강 데이터를 합쳐서 새로운 훈련 세트 만들기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)

# 라벨을 원-핫 인코딩으로 변환
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(16, 2, 1, input_shape=(28, 28, 1)))  # 2x2 필터 16개
model.add(Conv2D(8, 3, 1))                            # 3x3 필터 8개
model.add(MaxPooling2D())                             # 풀링으로 크기 축소
model.add(Conv2D(8, 3, 1))                            # 또 3x3 필터 8개
model.add(MaxPooling2D())                             # 풀링
model.add(Flatten())                                  # 1차원으로 펼침
model.add(Dense(units=32, activation='relu'))         # 완전연결층 32개 뉴런
model.add(Dropout(0.3))                               # 30% 드롭아웃
model.add(Dense(units=16, activation='relu'))         # 완전연결층 16개 뉴런
model.add(Dropout(0.3))                               # 30% 드롭아웃
model.add(Dense(units=10, activation='softmax'))      # 출력층 (10개 클래스)

model.summary()

# 3. 컴파일과 훈련 설정
model.compile(
    loss='categorical_crossentropy',  # 다중 클래스 분류 손실함수
    optimizer='adam',                 # 최적화 알고리즘
    metrics=['acc']                   # 정확도 확인
)

# 조기 종료 콜백: val_loss가 10번 연속 개선 안되면 종료
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()  # 학습 시작 시간 기록

model.fit(
    x_train, y_train,
    epochs=5000,                 # 최대 반복 횟수
    batch_size=256,              # 배치 크기
    validation_split=0.2,        # 검증 데이터 20%
    callbacks=[es],              # 조기 종료 사용
    verbose=2                    # 학습 로그 출력 형식
)

end_time = time.time()  # 학습 종료 시간 기록

# 4. 평가와 예측
results = model.evaluate(x_test, y_test)  # 테스트 데이터 평가
print('loss : ', results[0])               # 손실 출력
print('acc : ', results[1])                # 정확도 출력
print(end_time - start_time)               # 학습에 걸린 시간 출력

# augment 전 결과 (참고)
# loss :  0.45550528168678284
# acc :  0.8575000166893005
# 147.2709023952484
