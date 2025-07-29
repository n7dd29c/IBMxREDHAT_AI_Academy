import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)             # python 랜덤 고정
np.random.seed(SEED)          # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#1. 데이터
# datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 이 변수는 현재 코드에서 사용되지 않습니다.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape)     # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)

# y의 dtype을 torch.float32로 유지하고, loss 계산 시 차원 일치를 위해 reshape 필요
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE).unsqueeze(1) # (7,) -> (7,1)로 차원 추가
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7, 1])

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=1,        # feature(열)의 갯수, tensorflow에선 input_dim
            hidden_size=32,      # output_node의 갯수, tensorflow에선 unit
            # num_layers=1,      # default, RNN은닉층 레이어의 갯수
            batch_first=True,    # default는 false
            # 원래 (N, 3, 1)인데 False를 쓰면 (3, N, 1)
            # 그래서 다시 True쓰면 원위치 됨, 그냥 True쓰자
            # (N, 3, 32)
        )
        self.rnn_layer2 = nn.RNN(32, 20, batch_first=True)
        # self.fc1 = nn.Linear(3*32, 16)
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn_layer1(x)
        x = self.relu(x)
        x, _ = self.rnn_layer2(x)
        x = self.relu(x)
        
        # RNN 출력을 flatten
        # x = x.reshape(-1, 3*32)
        x = x[:, -1, :] # flatten 역할, 과적합 위험을 줄여준다
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary
summary(model, (3, 1))

# 손실 함수를 MSELoss로 변경
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.04)

def train(model, criterion, optimizer, loader):
    model.train()
    epochs_loss = 0
    epochs_r2 = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad() # 각 배치마다 기울기 초기화
        
        hypothesis = model(x_batch) # 모델 예측
        loss = criterion(hypothesis, y_batch) # 손실 계산
        
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        
        # r2_score 계산 (hypothesis 사용)
        y_pred_np = hypothesis.cpu().detach().numpy() # GPU -> CPU -> numpy
        y_batch_np = y_batch.cpu().detach().numpy()
        
        # r2_score를 위해 1차원으로 flatten
        r2 = r2_score(y_batch_np.flatten(), y_pred_np.flatten()) 
        
        epochs_loss += loss.item()
        epochs_r2 += r2

    return epochs_loss / len(loader), epochs_r2 / len(loader)

def evaluate(model, criterion, loader):
    model.eval() # 모델을 평가 모드로 설정
    epochs_loss = 0
    epochs_r2 = 0
    
    with torch.no_grad(): # 기울기 계산 비활성화
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch) # 모델 예측
            loss = criterion(hypothesis, y_batch) # 손실 계산
            
            # r2_score 계산 (hypothesis 사용)
            y_pred_np = hypothesis.cpu().numpy() # GPU -> CPU -> numpy
            y_batch_np = y_batch.cpu().numpy()
            
            # r2_score를 위해 1차원으로 flatten
            r2 = r2_score(y_batch_np.flatten(), y_pred_np.flatten()) 
            
            epochs_loss += loss.item()
            epochs_r2 += r2

    return epochs_loss / len(loader), epochs_r2 / len(loader)

epochs = 100
print("\n--- 학습 시작 ---")
for epoch in range(1, epochs + 1):
    loss, r2_train = train(model, criterion, optimizer, train_loader)
    val_loss, val_r2 = evaluate(model, criterion, train_loader) # 여기서는 훈련셋으로 평가
    print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | R2_train: {r2_train:.4f} | Val_Loss: {val_loss:.4f} | Val_R2: {val_r2:.4f}")