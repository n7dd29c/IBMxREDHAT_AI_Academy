import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

x = np.array([range(10), range(21, 31), range(201, 211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]]).T

print(x.shape, y.shape) # (10, 3) (10, 2)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

print(x.dtype)          # torch.float64
print(x.shape, y.shape) # torch.Size([10, 3]) torch.Size([10, 2])

# Standerd Scaler
x_scaled = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 6),
    nn.Linear(6, 7),
    nn.Linear(7, 5),
    nn.Linear(5, 2),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epoch = 1000
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

x_pred = (torch.Tensor([[10, 31, 211]]).to(DEVICE) - torch.mean(x)) / torch.std(x)

result = model(x_pred)

print(f'x_predict : {result.detach()}')
print(f'x_predict : {result.detach().cpu().numpy()}')
