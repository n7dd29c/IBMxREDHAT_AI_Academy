import numpy as np
import sklearn as sk
import pandas as pd
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [652 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [116 rows x 8 columns]
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)   # [116 rows x 1 columns]

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

y = train_csv['Outcome']
print(x)        # [652 rows x 8 columns]
print(y.shape)  # (652,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=1998,
)

#2. 모델구성

#3. 컴파일, 훈련
modelpath = './_save/keras28_mcp/07_dacon_diabetes/'
model = load_model(modelpath + 'k28_250604_$H38_0853-0.6418.hdf5')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict)
print('acc : ', acc_score)

# loss :  0.5966358184814453
# acc :  0.7424242424242424

# loss :  0.5966358184814453
# acc :  0.7424242424242424