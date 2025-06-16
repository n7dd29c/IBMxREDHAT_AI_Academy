import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터

np_path = 'c:/study25/_data/_save_npy/'
submit_path = './_data/kaggle/cat_dog/'
x = np.load(np_path + 'keras44_01_x_train_catdog.npy')
y = np.load(np_path + 'keras44_01_y_train_catdog.npy')
x_sub = np.load(np_path + 'keras44_01_x_sub_catdog.npy')

submission_csv = pd.read_csv(submit_path + 'sample_submission.csv')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, 
)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, 3, 1, input_shape=(100,100,3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(256, 3, 1, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

path = './_save/keras45/'
filename = '{epoch:04d}_{val_acc:.4f}.hdf5'
filepath = ''.join([path, 'k45_catdog_', filename])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath
)

start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=64, callbacks=[es,mcp],
          validation_split=0.2, verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results[0])           # 0.5785045623779297
print(results[1])           # 0.7361999750137329
y_pred = model.predict(x_sub)
submission_csv['label'] = y_pred
submission_csv.to_csv(submit_path + 'submission_0613.csv', index=False)
print(end_time-start_time)  # 845.2242391109467

# 0.5124953985214233
# 0.7495999932289124
# 3199.773459672928