import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torchvision.transforms as tr
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

transf = tr.Compose([tr.Resize(56), tr.ToTensor()])

path = './Study25/_data/torch/'
train_dataset = FashionMNIST(path, train=True, download=True, transform=transf)
test_dataset = FashionMNIST(path, train=False, download=True, transform=transf)

print(train_dataset)
print(type(train_dataset))  # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])

img_tensor, label = train_dataset[0]    # tuple 형태의 데이터는 이렇게 나눌 수 있다
print(label)                            # 9
print(img_tensor.shape)                 # torch.Size([1, 56, 56])
print(img_tensor.min(),                 # tensor(0.)
      img_tensor.max())                 # tensor(0.9765)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
print(len(train_loader))                # 1875

class CNN(nn.Module):
    def __init__(self, num_featres):
        # super().__init__()
        super(CNN, self).__init__() # 위에랑 같은 말
        
        self.hidden_layer1 = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3),
            nn.ReLU()
        )
    
        self.hidden_layer2 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        
        self.hidden_layer4 = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        self.hidden_layer5 = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.flatten(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = CNN(1).to(DEVICE)

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

# final_loss : 0.4151, final_acc : 0.8528