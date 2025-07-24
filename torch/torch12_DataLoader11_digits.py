import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_digits()
x = datasets.data
y = datasets.target
print(datasets.target_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.int64).to(DEVICE) # long과 int64는 같다

print(x_train.shape, x_test.shape)  # torch.Size([120, 4]) torch.Size([30, 4])

# model = nn.Sequential(
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 10),
#     # nn.Softmax(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()                      # 아래와 같은 결과
        # super(Model, self).__init__()           # nn.Module에 있는 Model과 self 다 쓰겠다
        
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.soft(x)
        return x
    
model = Model(64,10).to(DEVICE)

criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epochs : {epoch}, loss : {loss}')
    
def evaluate(model, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        y_pred_class = torch.argmax(y_pred, dim=1) # 가장 높은 확률을 가진 클래스 선택
        
        # 정확도 계산
        accuracy = accuracy_score(y.detach().cpu().numpy(), y_pred_class.detach().cpu().numpy())
    return accuracy, y_pred # 정확도와 원시 예측값 모두 반환

# 최종 평가
accuracy, y_predict_raw = evaluate(model=model, x=x_test, y=y_test)
print('Accuracy :', accuracy)