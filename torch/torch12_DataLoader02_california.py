import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(datasets.target_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape)  # torch.Size([120, 4]) torch.Size([30, 4])

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


###################### 모델에 대한 정의 ######################

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()                      # 아래와 같은 결과
        # super(Model, self).__init__()           # nn.Module에 있는 Model과 self 다 쓰겠다
        
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, 4)
        self.linear6 = nn.Linear(4, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        return x
    
model = Model(8,1).to(DEVICE)
        
##############################################################

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    
    for x_batch, y_batch in loader:
        model.train()
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, loader=train_loader)
    print(f'epochs : {epoch}, loss : {loss}')
    
def evaluate(model, criterion, loader):
    model.eval()
    total_loader = 0
    
    total_loss = 0
    for x_batch, y_batch in loader:
        
        with torch.no_grad():
            y_pred = model(x_batch)
            final_loss = criterion(y_batch, y_pred)
            
            total_loss += final_loss.item()
            
    return total_loss / len(loader)

# 최종 평가
final_loss = evaluate(model=model, criterion=criterion, loader=test_loader)
print('final_loss :', final_loss)

y_pred = model(x_test)
r2 = r2_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
print(r2)