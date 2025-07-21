import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch :", torch.__version__, "사용 DEVICE :", DEVICE)
# torch : 2.7.1+cu128 사용 DEVICE : cuda

#1. 데이터
x = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]).transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# print(x.size())
# torch.Size([10, 2])
# exit()

########### Standard Scaler ###########
x_scaled = (x - torch.mean(x)) / torch.std(x)
#######################################

#2. 모델구성
# model = nn.Linear(1,1).to(DEVICE)
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()      # torch를 

#3. 컴파일 훈련    
epochs = 1000
for epoch in range(1, epochs+1) :
    loss = train(model=model, criterion=criterion, optimizer=optimizer, x=x_scaled, y=y)
    print(f'epoch: {epoch}, loss: {loss}')

#4. 평가 예측
print('=====================================')
def evaluate(model, criterion, x, y) :
    model.eval()
    # [평가모드] 드랍아웃, 배치노말 절대 안 됨
    
    # with torch.no_grad() : # 기울기 갱신을 하지 않겠다.
    #     y_pred = model(x)
    #     final_loss = criterion(y, y_pred)   # loss의 최종값

    torch.set_grad_enabled(False)    
    y_pred = model(x)
    final_loss = criterion(y, y_pred)   # loss의 최종값
    torch.set_grad_enabled(True)    
    
    # 위에 두개 중에 하나 골라서 쓰자
    
    return final_loss.item()

final_loss = evaluate(model=model, criterion=criterion, x=x_scaled, y=y)
print(f"최종 loss : {final_loss}")

x_pred = (torch.Tensor([[10, 1.3]]).to(DEVICE) - torch.mean(x)) / torch.std(x)

result = model(x_pred)

print(f"[10, 1.3]의 예측값 : {result}")
print(f"[10, 1.3]의 예측값 : {result.item()}")
# [10, 1.3]의 예측값 : 9.965142250061035







