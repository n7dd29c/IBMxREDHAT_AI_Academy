import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_pred = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape) # torch.Size([7, 1]) torch.Size([7, 1])
print(x_test.shape, y_test.shape)   # torch.Size([4, 1]) torch.Size([4, 1])
print(x_pred.shape)                 # torch.Size([3])

model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 6),
    nn.Linear(6, 7),
    nn.Linear(7, 4),
    nn.Linear(4, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 3000
for epochs in range(1, epochs+1):
    loss = train(model=model, criterion=criterion, optimizer=optimizer,
                 x=x_train, y=y_train)
    print(f'(train) epochs : {epochs}, loss : {loss}')
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
    return final_loss.item()

final_loss = evaluate(model=model, criterion=criterion, x=x_test, y=y_test)
print(f'final_loss : {final_loss}')

result = model(x_pred)

print(f'x_predict : {result.detach().flatten()}')   # 1개짜리면 .item, 2개이상이면 .detach
print(f'x_predict : {result.detach().cpu().numpy().flatten()}')

# final_loss : 1.7018055586959235e-07
# x_predict : tensor([12.0005, 13.0006, 14.0006], device='cuda:0')
# x_predict : [12.000519  13.000565  14.0006075]