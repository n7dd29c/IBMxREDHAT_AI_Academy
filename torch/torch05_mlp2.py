import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]]).T
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.dtype)          # float64
print(x.shape, y.shape) # (10, 3) (10,)

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)                  # 옛날 문법

# x = torch.FloatTensor(x, dtype=float).to(DEVICE)                  # 형변환
# y = torch.FloatTensor(y, dtype=float).unsqueeze(1).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)                 # torch.tensor 로 권장
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)    # 이 방법을 썼을 땐 형변환

# Standerd Scaler
x_scaled = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 6),
    nn.Linear(6, 7),
    nn.Linear(7, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epoch = 400
for epoch in range(1, epoch+1):
    loss = train(model=model, criterion=criterion, optimizer=optimizer, x=x_scaled, y=y)
    print(f'epoch : {epoch}, loss : {loss}')
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
        
        return final_loss.item()
    
final_loss = evaluate(model=model, criterion=criterion, x=x_scaled, y=y)
print(f'final_loss : {final_loss}')

x_pred = (torch.Tensor([[10, 1.3, 0.5]]).to(DEVICE) - torch.mean(x)) / torch.std(x)

result = model(x_pred)

print(f'x_predict : {result.item()}')

# final_loss : 0.8280102014541626
# x_predict : 8.229192733764648