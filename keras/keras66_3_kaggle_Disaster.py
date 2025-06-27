import numpy as np
import pandas as pd
import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ran = 656

#1. 데이터
path = './_data/kaggle/Disaster/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + 'keyword: ' + df['keyword'] + 'location: ' + df['location']
    return df

train = merge_text(train_csv)
test = merge_text(test_csv)

x = train['text']
y = train['target']
test_text = test['text']

print(train)    # [7613 rows x 3 columns]
print(test)     # [3263 rows x 3 columns]
print(x.shape)  # (7613, 3)
print(y.shape)  # (7613,)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x)
size = len(tokenizer.word_index)
print(len(tokenizer.word_index))    # 25745

x_sequences = tokenizer.texts_to_sequences(x)
test_sequences = tokenizer.texts_to_sequences(test_text)

# 토큰화된 시퀀스들의 길이
lengths = [len(s) for s in x_sequences]

# 길이 분포 확인
print("최소 길이:", np.min(lengths))
print("최대 길이:", np.max(lengths))
print("평균 길이:", np.mean(lengths))
print("중앙값 (50th percentile) 길이:", np.median(lengths))
print("90th percentile 길이:", np.percentile(lengths, 90))
print("95th percentile 길이:", np.percentile(lengths, 95))
print("99th percentile 길이:", np.percentile(lengths, 99))

# 시각화 (선택 사항)
# import matplotlib.pyplot as plt
# plt.hist(lengths, bins=50)
# plt.title('Distribution of Sequence Lengths')
# plt.show()
# exit()

from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(
                x_sequences, 
                padding='pre',      # 'post'는 뒤로, default : pre
                maxlen=27,
                truncating='pre'    # 'post'는 뒤로, default : pre
)

padding_test = pad_sequences(
    test_sequences,
    maxlen=27,
    padding='post',
    truncating='post'
)

print(padding_x)
print(padding_x.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    padding_x, y, random_state=ran, test_size=0.2
)

print(x_train.shape, x_test.shape)  # (6090, 3) (1523, 3)
print(y_train.shape, y_test.shape)  # (6090,) (1523,)

#2. 모델구성
model = Sequential()
model.add(Embedding(input_dim=size, output_dim=16, input_length=27))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(32, input_dim=3))
model.add(Dropout(0.2))
model.add(Dense(16, input_dim=3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=2
)

save_path = './_data/kaggle/Disaster/save/'
filepath = ''.join([save_path, 'k66_', datetime.datetime.now().strftime('%m%d_%M'),
                    '{epoch:04d}_{val_loss:.4f}.hdf5'])
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath,
    verbose=2
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, verbose=2, callbacks=[es,mcp])
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("Loss:", loss)
print("Accuracy:", acc)

y_predict = model.predict(padding_test)
y_submit = np.where(y_predict > 0.5, 1, 0)
print("훈련 시간:", round(end_time, 2), "초")

submission_csv['target'] = y_submit
submission_csv.to_csv(save_path + 'k66_' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.csv')

print("제출 파일 생성 완료!")