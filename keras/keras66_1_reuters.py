from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, # 단어사전의 갯수, 빈도수가 높은 단어 순으로 1000개 뽑기
    test_split=0.2,
    # maxlen=100,     # 단어 길이가 100개까지 있는 문장, 100개 이상은 자름
)

print(x_train)
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)
print(y_train[0])                   # 3
print(np.unique(y_train))           # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                                    #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
                  
print(type(x_train))                # <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'>
print(max(len(i) for i in x_train)) # 2376, 뉴스기사의 최대길이
print(min(len(i) for i in x_train)) # 13, 뉴스기사의 최소길이
print(sum(map(len, x_train))/len(x_train))  # 145.5398574927633, 뉴스기사의 평균길이

padding_train = pad_sequences(
    x_train,
    padding='pre',
    maxlen=10,
    truncating='pre'
)
padding_test = pad_sequences(x_test, maxlen=10)

y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)

print(padding_train)
print(padding_train.shape)              # (8982, 10)

model = Sequential()
model.add(Embedding(1000, 512))
model.add(LSTM(512))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
model.fit(padding_train, y_train, epochs=100, batch_size=16 , callbacks=es, verbose=2)

loss = model.evaluate(padding_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])
