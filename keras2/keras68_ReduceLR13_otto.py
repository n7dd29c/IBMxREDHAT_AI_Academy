import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.optimizers import Adam

# 1. 데이터 로드
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(f"Original train_csv shape: {train_csv.shape}")
print(f"Original test_csv shape: {test_csv.shape}")

# 2. x, y 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']  # 범주형 문자열 (예: Class_1 ~ Class_9)

# 3. y 인코딩 (문자열 → 정수 → One-Hot)
le = LabelEncoder()
y_encoded = le.fit_transform(y)               # 예: Class_1 → 0, ..., Class_9 → 8
y_ohe = to_categorical(y_encoded)             # One-hot 인코딩

# 4. train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, test_size=0.2, random_state=55, stratify=y_encoded
)

# 5. 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 6. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=93, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(9, activation='softmax'))  # 클래스 수에 맞게 설정

# 7. 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# 콜백
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=2, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=2, factor=0.5)

# 8. 훈련
start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es, rlr],
    verbose=0
)
end_time = time.time()

# 9. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test, verbose=2)
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

final_acc = accuracy_score(y_true, y_pred)
training_time = end_time - start_time

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {final_acc:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
