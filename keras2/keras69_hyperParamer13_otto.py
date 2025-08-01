import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')

path = './_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(f"Original train_csv shape: {train_csv.shape}") # (61878, 94) - Otto 데이터셋 기준
print(f"Original test_csv shape: {test_csv.shape}")   # (144368, 93) - Otto 데이터셋 기준

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y # stratify는 원본 y 사용
)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001) :
    inputs = Input(shape=(93,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

def create_hyperparameter():
    batch = [32,16,8,1,64]
    optimizer = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'softmax']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    node4 = [128,64,32,16]
    node5 = [128,64,32,16,8]
    return {
        'batch_size' : batch,
        'optimizer' : optimizer,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }

hyper = create_hyperparameter()
from scikeras.wrappers import KerasRegressor

import time
keras_model = KerasRegressor(
    model=build_model,
    verbose=1,
    optimizer="adam",
    drop=0.5,
    activation="relu",
    node1=128,
    node2=64,
    node3=32,
    node4=16,
    node5=8,
    lr=0.001,
    batch_size=32,
    epochs=100,
    
)

model = RandomizedSearchCV(keras_model, hyper, cv=2, n_iter=2)
s_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=-1,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=10,
    verbose=-1,
    factor=0.8,
)
model.fit(x_train, y_train, callbacks=[es,rlr], validation_split=0.2,)
e_time = time.time()

# print('    best_variable :', model.best_estimator_)
print('      best_params :', model.best_params_)

#4. 평가
print('       best_score :', model.best_score_)
print(' model_best_score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('   accuracy_score :', acc)
print('     running_time :', np.round(e_time - s_time, 3), 'sec')



