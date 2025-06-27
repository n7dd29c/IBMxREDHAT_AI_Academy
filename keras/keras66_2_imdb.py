from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,
)
print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))
print(pd.value_counts(y_train))
print('영화평의 최대길이 : ', max(len(i) for i in x_train))         # 2494
print('영화평의 최소길이 : ', min(len(i) for i in x_train))         # 11
print('영화평의 평균길이 : ', sum(map(len, x_train))/len(x_train))  # 238.71364

from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_train = pad_sequences(
    x_train,
    padding='pre',
    maxlen=10,
    truncating='pre'
    )
padding_test = pad_sequences(x_test, maxlen=10)

