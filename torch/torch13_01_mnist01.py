import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

path = './Study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

print(train_dataset)
print(type(train_dataset))  # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])     # (<PIL.Image.Image image mode=L size=28x28 at 0x76E11B193730>, 5)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets
# print(x_train, y_train)
print(x_train.shape, y_train.size())    # torch.Size([60000, 28, 28]) torch.Size([60000])
print(np.min(x_train.numpy()), np.max(x_train.numpy()))     # 0 1

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 28*28)
print(x_train.shape, y_train.size())    # torch.Size([60000, 784]) torch.Size([60000])

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class DNN(nn.Module):
    def __init__(self, num_features):
        # super().__init__()
        super(DNN, self).__init__() # 위에랑 같은 말
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU()
        )
    
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = DNN(784).to(DEVICE)

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
