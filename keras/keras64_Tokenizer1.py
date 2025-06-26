import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

text = '오늘도 못생기고 영어를 디게 디게 디게 못하는 이삭이는 \
        재미없는 개그를 마구 마구 마구 하면서 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'디게': 1, '마구': 2, '오늘도': 3, '못생기고': 4, '영어를': 5, '못하는': 6, '이삭이는': 7, '재미없는': 8, '개그를': 9, '하면서': 10, '딴짓을': 11, '한다': 12}
print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 3), ('못하는', 1), ('이삭이는', 1), ('재미없는', 1), ('개그를', 1), ('마구', 3), ('하면서', 1), ('딴짓을', 1), ('한다', 1)]) 

x = token.texts_to_sequences([text])
print(x)
# [[3, 4, 5, 1, 1, 1, 6, 7, 8, 9, 2, 2, 2, 10, 11, 12]]

############################ OneHot 3가지 만들기 ############################

#1. pandas
x_1 = pd.get_dummies(np.array(x).reshape(-1,))
print(x_1)
print(x_1.shape)

#2. sklearn
from sklearn.preprocessing import OneHotEncoder
x_2 = np.reshape(x, (-1,1))
ohe = OneHotEncoder(sparse_output=False)
x_2 = ohe.fit_transform(x_2)
print(x_2)
print(x_2.shape)

#3. keras
from tensorflow.keras.utils import to_categorical
x_3 = to_categorical(x)
print(x_3)
x_3 = x_3[:, :, 1:]
x_3 = x_3.reshape(16, 12)
print(x_3.shape)
