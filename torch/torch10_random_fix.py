import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import random

###################### 랜덤 고정 ######################
SEED = 337
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.dtype)
print(type(x_train))
print(x_train.shape, y_train.shape)
# torch.float32
# <class 'torch.Tensor'>
# torch.Size([455, 30]) torch.Size([455, 1])

model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epochs in range(1, epochs+1):
    loss = train(model=model, criterion=criterion, optimizer=optimizer, x=x_train, y=y_train)
    print(f'epochs : {epochs}, loss : {loss}')
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
    return final_loss

final_loss = evaluate(model=model, criterion=criterion, x=x_test, y=y_test)
print('final :', final_loss)

y_predict = model(x_test)
y_predict = y_predict.detach()
print(type(y_predict))  # <class 'torch.Tensor'>
y_predict = y_predict.cpu()
print(type(y_predict))  # <class 'torch.Tensor'>
y_predict = y_predict.numpy()
print(type(y_predict))  # <class 'numpy.ndarray'>
y_predict = np.round(y_predict)
# 이 일련의 과정을

y_test = y_test.detach().cpu().numpy()
# 위 처럼 줄일 수 있다

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)     # acc : 0.9736842105263158
