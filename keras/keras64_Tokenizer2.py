import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 이삭이는 \
        재미없는 개그를 마구 마구 마구 마구 하면서 딴짓을 한다.'
        
text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. \
        상진이는 마구 마구 딴짓을 한다. 재현은 못생기고 재미없는 딴짓을 한다.'
        
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'디게': 1, '마구': 2, '딴짓을': 3, '한다': 4, '오늘도': 5, '못생기고': 6, '재미없는': 7,
#  '영어를': 8, '못하는': 9, '이삭이는': 10, '개그를': 11, '하면서': 12, '박석사가': 13, '자아를': 14, '찾아냈다
# ': 15, '상진이는': 16, '재현은': 17}

print(token.word_counts)
# OrderedDict([('오늘도', 2), ('못생기고', 2), ('영어를', 1), ('디게', 5), ('못하는', 1), ('이삭이는', 1),
#              ('재미없는', 2), ('개그를', 1), ('마구', 5), ('하면서', 1), ('딴짓을', 3), ('한다', 3),
#              ('박석사가', 1), ('자아를', 1), ('찾아냈다', 1), ('상진이는', 1), ('재현은', 1)])

x = token.texts_to_sequences([text1, text2])
print(x)
# [[5,  6,  8, 2, 2,  2,  9, 10, 11, 7, 12,  1, 1, 1, 1, 13, 3, 4],
#  [5, 14, 15, 2, 2, 16, 17,  1,  1, 3,  4, 18, 6, 7, 3,  4]]

x = np.array(x)
print(x)
# [list([5, 6, 8, 2, 2, 2, 9, 10, 11, 7, 12, 1, 1, 1, 1, 13, 3, 4])
#  list([5, 14, 15, 2, 2, 16, 17, 1, 1, 3, 4, 18, 6, 7, 3, 4])]

x = np.concatenate(x)
print(x)
# [ 5  6  8  2  2  2  9 10 11  7 12  1  1  1  1 13  3  4  5 14 15  2  2 16
#  17  1  1  3  4 18  6  7  3  4]

#1. pandas
x_1 = pd.get_dummies(np.array(x).reshape(-1,))
print(x_1)
print(x_1.shape)    # (34, 18)

#2. sklearn
from sklearn.preprocessing import OneHotEncoder
x_2 = np.reshape(x, (-1,1))
ohe = OneHotEncoder(sparse_output=False)
x_2 = ohe.fit_transform(x_2)
print(x_2)
print(x_2.shape)    # (34, 18)

#3. keras
from tensorflow.keras.utils import to_categorical
x_3 = to_categorical(x)
print(x_3)
x_3 = x_3[:, 1:]
x_3 = x_3.reshape(34, 18)
print(x_3.shape)
