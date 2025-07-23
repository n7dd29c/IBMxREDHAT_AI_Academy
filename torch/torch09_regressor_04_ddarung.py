import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

path = './Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)        # [1459 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)         # [715 rows x 9 columns]

##################### 결측치 처리 2. 평균치 넣기 ####################
train_csv = train_csv.fillna(train_csv.mean())

#################### 결측치 처리 3. 테스트 데이터 ###################
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)   # drop() : 행 또는 열 삭제
                                        # count라는 열(axis=1) 삭제, 참고로 행은 axis=0
# print(x)                                # [1459 rows x 9 columns]

y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55,
)

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape)

model = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    # nn.Softmax(),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epochs : {epoch}, loss : {loss}')
    
def evaluate(model, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        # y_pred_class = torch.argmax(y_pred, dim=1) # 가장 높은 확률을 가진 클래스 선택
        
        # 정확도 계산
        r2 = r2_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    return r2, y_pred # 정확도와 원시 예측값 모두 반환

# 최종 평가
r2, y_predict_raw = evaluate(model=model, x=x_test, y=y_test)
print('r2 :', r2)