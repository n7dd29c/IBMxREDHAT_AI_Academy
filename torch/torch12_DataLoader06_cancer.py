import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

###################### 랜덤 고정 ######################
SEED = 337
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=SEED
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

################################ torch 데이터셋 만들기 ################################
from torch.utils.data import TensorDataset      # x, y 합치기
from torch.utils.data import DataLoader         # batch 정의

#1. x, y 합치기
train_set = TensorDataset(x_train, y_train)     # tuple 형태
test_set = TensorDataset(x_test, y_test)        # 마찬가지
print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x7fc562044af0>
print(type(test_set))       # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))       # 512
print(train_set[0])
# (tensor([ 1.4965, -0.1038,  1.5086,  1.5481,  1.3658,  1.1458,  1.6450,  1.4644,
#         -0.0656,  0.4029,  2.4621, -0.1111,  2.0842,  1.9490, -0.3405,  0.3761,
#          0.8901,  0.9287, -0.2161,  0.3825,  1.9450,  0.0761,  1.8511,  1.9914,
#          0.7708,  1.0288,  1.7138,  1.2709,  0.1072,  0.9026], device='cuda:0'),  <- x
#                                                   tensor([0.], device='cuda:0'))  <- y
print(train_set[0][0])      # 첫번째 x
print(train_set[0][1])      # 첫번째 y

#2. batch 정의
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)   # test는 False
print(len(train_loader))    # 16
print(train_loader) # <torch.utils.data.dataloader.DataLoader object at 0x75c105dc36a0>
# print(train_loader[0][0]) # 이런 방법으로는 형태를 확인할 수 없다
print('\n=====================================================================================\n')

#---------------- iterator로 확인하기 ----------------#
#1. for문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break           # 첫번째 batch 출력

for x_batch, y_batch in train_loader:
    print(x_batch, y_batch)
    break

# #2. next() 사용
# bbb = iter(train_loader)
# # aaa = bbb.next()  # 파이썬 버전이 올라가고 없어짐
# aaa = next(bbb)
# print(aaa)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)  
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
    
model = Model(30,1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

epochs = 1000
for epochs in range(1, epochs+1):
    loss = train(model=model, criterion=criterion, optimizer=optimizer, loader=train_loader)
    print(f'epochs : {epochs}, loss : {loss}')
    
def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        
        with torch.no_grad():
            y_pred = model(x_batch)
            final_loss = criterion(y_batch, y_pred)
            
            total_loss += final_loss.item()
            
    return total_loss / len(loader)

final_loss = evaluate(model=model, criterion=criterion, loader=test_loader)
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
# 처럼 줄일 수 있다

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)     # acc : 0.9736842105263158
