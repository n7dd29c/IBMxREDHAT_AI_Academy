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
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape)     # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE).unsqueeze(1) # (7,) -> (7,1)로 차원 추가
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7, 1])

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True) # 배치 크기 2

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=1,         # feature(열)의 갯수, tensorflow에선 input_dim
            hidden_size=32,       # output_node의 갯수, tensorflow에선 unit
            batch_first=True,     
            bidirectional=True,  
        )
        self.fc1 = nn.Linear(32 * 2, 16) 
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU() 
        
    def forward(self, x):
        x, _ = self.rnn_layer1(x) 
        x = self.relu(x)
        
        x = x[:, -1, :] 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary
summary(model, (3, 1))

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    model.train()
    epochs_loss = 0
    epochs_r2 = 0
    num_batches_for_r2 = 0 # R2가 정상적으로 계산된 배치의 수를 세기 위함
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad() 
        
        hypothesis = model(x_batch) 
        loss = criterion(hypothesis, y_batch) 
        
        loss.backward() 
        optimizer.step() 
        
        y_pred_np = hypothesis.cpu().detach().numpy() 
        y_batch_np = y_batch.cpu().detach().numpy()
        
        y_batch_flat = y_batch_np.flatten()
        y_pred_flat = y_pred_np.flatten()
        
        # y_true의 고유한 값이 1개(즉, 모든 값이 동일)인 경우 R2_score가 정의되지 않음
        if len(np.unique(y_batch_flat)) > 1:
            r2 = r2_score(y_batch_flat, y_pred_flat) 
            epochs_r2 += r2
            num_batches_for_r2 += 1
        # else:
            # print(f"Warning: y_batch has constant values in a batch. R2 score skipped for this batch. y_batch: {y_batch_flat}")
        
        epochs_loss += loss.item()

    # R2 계산 시, R2가 실제로 계산된 배치의 수로 나눔
    avg_loss = epochs_loss / len(loader)
    avg_r2 = epochs_r2 / num_batches_for_r2 if num_batches_for_r2 > 0 else float('nan')
    
    return avg_loss, avg_r2

epochs = 500
print("\n--- 학습 시작 ---")
for epoch in range(1, epochs + 1):
    loss, r2_train = train(model, criterion, optimizer, train_loader)
    print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | R2_train: {r2_train:.4f}")

def evaluate(model, loader):
    model.eval() 
    epochs_loss = 0
    epochs_r2 = 0
    num_batches_for_r2 = 0 # R2가 정상적으로 계산된 배치의 수를 세기 위함
    
    with torch.no_grad(): 
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch) 
            loss = criterion(hypothesis, y_batch) 
            
            y_pred_np = hypothesis.cpu().numpy() 
            y_batch_np = y_batch.cpu().numpy()
            
            y_batch_flat = y_batch_np.flatten()
            y_pred_flat = y_pred_np.flatten()
            
            if len(np.unique(y_batch_flat)) > 1:
                r2 = r2_score(y_batch_flat, y_pred_flat) 
                epochs_r2 += r2
                num_batches_for_r2 += 1
            # else:
                # print(f"Warning: y_batch has constant values in evaluation batch. R2 score skipped. y_batch: {y_batch_flat}")
            
            epochs_loss += loss.item()

    avg_loss = epochs_loss / len(loader)
    avg_r2 = epochs_r2 / num_batches_for_r2 if num_batches_for_r2 > 0 else float('nan')
    
    return avg_loss, avg_r2

# 최종 평가
final_loss, final_r2 = evaluate(model=model, loader=train_loader)
print(f'Final Loss: {final_loss:.4f}, Final R2: {final_r2:.4f}')

# --- 새로운 데이터 예측 ---
input_data_np = np.array([8,9,10])
input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).reshape(1, -1, 1).to(DEVICE)

model.eval()
with torch.no_grad():
    y_pred_tensor = model(input_data_tensor)

y_pred_value = y_pred_tensor.detach().cpu().numpy().flatten()[0]

print(f'Prediction for [8,9,10]: {y_pred_value:.4f}')