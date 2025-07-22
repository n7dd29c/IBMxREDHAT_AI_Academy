import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')    # 'cuda:0' 과 똑같은 말

x = np.array(range(100))
y = np.array(range(1, 101))
x_pred = np.array([101, 102])

x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# x_train = x[:-20]
# x_test = x[-20:]
# print(x_train)
# print(x_test)

# y_train = y[:-20]
# y_test = y[-20:]
# print(y_train)
# print(y_test)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=33
)
# exit()
x_train_scale = (x_train - torch.mean(x_train)) / torch.std(x_train)
x_test_scale = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_pred_scale = (x_pred- torch.mean(x_train)) / torch.std(x_train)

model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 6),
    nn.Linear(6, 7),
    nn.Linear(7, 4),
    nn.Linear(4, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

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
                 x=x_train_scale, y=y_train)
    print(f'(train) epochs : {epochs}, loss : {loss}')
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
    return final_loss.item()

final_loss = evaluate(model=model, criterion=criterion, x=x_test_scale, y=y_test)
print(f'final_loss : {final_loss}')

result = model(x_pred_scale)

print(f'x_predict : {result.detach().flatten()}')
print(f'x_predict : {result.detach().cpu().numpy().flatten()}')

# final_loss : 2.096669539375373e-11
# x_predict : tensor([102.0000, 103.0000], device='cuda:0')
# x_predict : [102.      103.00001]