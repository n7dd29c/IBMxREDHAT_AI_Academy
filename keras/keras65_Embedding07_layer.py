import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '석준이 바보', '준희 잘생겼다', '이삭이 또 구라친다',
]   # x data

label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0]) # y data

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9,
#  '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18,
#  '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '석준이': 24, '바보': 25,
#  '준희': 26, '잘생겼다': 27, '이삭이': 28, '또': 29, ' 구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16],
#  [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

################################ padding ################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(
                x, 
                padding='pre',      # 'post'는 뒤로, default : pre
                maxlen=5,
                truncating='pre'    # 'post'는 뒤로, default : pre
)

print(padding_x)
print(padding_x.shape)              # (15, 5)

model = Sequential()

#region Embedding 1
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5, ))
#                     # input_dim : 단어사전의 갯수(말뭉치 갯수)
#                     # output_dim : 다음 레이어로 전달하는 노드의 갯수(조절가능)
#                     # input_length : (N, 5), 컬럼의 갯수, 문장 시퀀스의 갯수
# model.add(LSTM(16))
# model.add(Dense(1))
# model.summary()
#endregion

#region Embedding 2
# model.add(Embedding(input_dim=31, output_dim=100)) # input_length 없이도 자동으로 맞춰줌
# model.add(LSTM(16))
# model.add(Dense(1))
#endregion

#region Embedding 3
# model.add(Embedding(input_dim=13, output_dim=100))
# # input_dim을 많게하면 메모리 사용량이 많아짐
# # input_dim을 작게하면 단어사전을 줄여버려 성능이 저하
# model.add(LSTM(16))
# model.add(Dense(1))
#endregion

#region Embedding 4
model.add(Embedding(31, 100, input_length=5))
# model.add(Embedding(31, 100, input_length=1))   # input_length=1은 먹힘
# 기본이 되는 input_dim과 output_dim은 명시를 하지 않아도 되지만 다른건 명시해줘야 함
model.add(LSTM(16))
model.add(Dense(1))
#endregion

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padding_x, label, epochs=100, )

loss = model.evaluate(padding_x, label)
print('loss : ', loss[0])
print('acc : ', loss[1])

x_pred = ['이삭이 참 잘생겼다']
x_pred = token.texts_to_sequences(x_pred)
x_pred = pad_sequences(x_pred, maxlen=5)

results = model.predict(x_pred)
print(np.round(results))