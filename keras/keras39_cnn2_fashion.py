import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True))
# print(pd.value_counts(y_test))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(16, 2, 1, input_shape=(28, 28, 1)))    # filters, kernel_size, strides 생략
model.add(Conv2D(8, 3, 1))
model.add(MaxPooling2D())
model.add(Conv2D(8, 3, 1))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))
model.summary()
# model.add(Dense())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=256, validation_split=0.2,
          callbacks=[es], verbose=2)
end_time = time.time()

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print(end_time-start_time)

import matplotlib.pyplot as plt
plt.imshow(x_train[11], 'gray')
plt.show()

# loss :  0.45550528168678284
# acc :  0.8575000166893005
# 147.2709023952484