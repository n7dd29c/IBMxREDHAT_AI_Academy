import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    fill_mode='nearest'
)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)
print(np.max(randidx), np.min(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    -1, x_augmented.shape[1], x_augmented.shape[2], 1
)

x_augmented = datagen.flow(
    x_augmented,
    batch_size=augment_size,
    shuffle=False
).next()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())      
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))       # input_shape는 생략가능
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=256, validation_split=0.2,
          callbacks=[es], verbose=2)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print(end_time - start_time)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = y_test.values
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))

# loss :  0.20601223409175873
# acc :  0.9753999710083008

# loss :  0.23427651822566986
# acc :  0.9785000085830688
# 707.1984360218048