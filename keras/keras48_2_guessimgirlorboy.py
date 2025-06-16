import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

#1. 데이터

np_path = 'c:/study25/_data/_save_npy/'
x = np.load(np_path + 'keras46_01_x_train_gender.npy')
y = np.load(np_path + 'keras46_01_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, test_size=0.2, 
)

#2. 모델구성

#3. 컴파일, 훈련
model_path = './_save/keras46/'
model = load_model(model_path + 'k46_gender_0015_0.5943.hdf5')

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
path = './_data/image/me/'
img = np.load(path + 'keras47_me.npy')
y_pred = model.predict(img)
# y_pred = model.predict(img)

if y_pred[0][0] < 0.5:
    print('남자')
else:
    print('여자')