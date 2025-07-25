import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from sklearn.preprocessing import StandardScaler

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])
# ToTensor : torch.tensor로 바꾸기 + MinMaxScaler 적용
# Z_score Nomalization 정규화의 표준화?
# (x - 평균) / 표준편차
# (x - 0.5) / 0.5          위 식처럼 해야하는데 통상 평균 0.5, 표편 0.5로 계산하면
# -1 ~ 1 사이의 범위가 나오니 이미지 전처리에서는 0.5 0.5로 처리한다

path = './Study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(len(train_dataset))               # 60000
print(train_dataset[0][0])
print(train_dataset[0][1])              # 5

img_tensor, label = train_dataset[0]    # tuple 형태의 데이터는 이렇게 나눌 수 있다
print(label)                            # 5
print(img_tensor.shape)                 # torch.Size([1, 56, 56])
print(img_tensor.min(),                 # tensor(0.)
      img_tensor.max())                 # tensor(0.9922)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
print(len(train_loader))                # 1875

class CNN(nn.Module):
    def __init__(self, num_features):
        # super().__init__()
        super(CNN, self).__init__() # 위에랑 같은 말
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=(3,3), stride=1),  # (1,56,56) -> (128,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                  # (n,128,27,27)
            nn.Dropout(0.2)
        )
    
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=1),            # (n,64,25,25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)                   # (n,64,12,12)
            
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),             # (n,32,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                  # (n,32,5,5)
        )
        
        self.flatten = nn.Flatten()
        
        self.hidden_layer4 = nn.Sequential(                             # Flatten에서 받아야 함
            nn.LazyLinear(16),       
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = CNN(1).to(DEVICE)   # torch에서는 channel만 input으로 넣어주면 된다, 나머지는 알아서

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)   #0.001

def train(model, criterion, optimizer, loader):

    epochs_loss = 0
    epochs_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward()
        optimizer.step()
        
        y_pred = torch.argmax(hypothesis, 1)
        acc = (y_pred == y_batch).float().mean()
        epochs_loss += loss.item()
        epochs_acc += acc

    return epochs_loss / len(loader), epochs_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
    epochs_loss = 0
    epochs_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean()
            
            epochs_loss += loss.item()
            epochs_acc += acc
        return epochs_loss / len(loader), epochs_acc / len(loader)

epochs = 100
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epochs : {epoch}, loss : {loss:.4f}, acc : {acc:.4f}, val_loss : {val_loss:.4f}, val_acc : {val_acc:.4f}')
    
loss, acc = evaluate(model, criterion, test_loader)
print(f'final_loss : {loss:.4f}, final_acc : {acc:.4f}')
