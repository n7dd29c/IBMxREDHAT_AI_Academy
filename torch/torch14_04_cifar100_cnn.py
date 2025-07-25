import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torchvision.transforms as tr
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)             # python 랜덤 고정
np.random.seed(SEED)          # numpy 랜덤 고정
torch.manual_seed(SEED)       # torch 랜덤 고정
torch.cuda.manual_seed(SEED)  # torch, cuda 시드 고정
# GPU를 사용하는 경우, 결정론적 알고리즘을 사용하도록 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(f"Using device: {DEVICE}") # 현재 사용 중인 디바이스 출력

path = './Study25/_data/torch/'

transf = tr.Compose([tr.ToTensor(),tr.Resize(64)])

# 데이터셋 로드 (transform 적용)
train_dataset = CIFAR100(root=path, train=True, download=True, transform=transf)
test_dataset = CIFAR100(root=path, train=False, download=True, transform=transf)

# DataLoader 생성:
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 데이터 로드 확인 및 shape 출력
# 첫 번째 배치 가져와서 실제 shape 확인
dataiter = iter(train_loader)
images, labels = next(dataiter)

print(f"첫 번째 배치 이미지 shape: {images.shape}") # 예시: torch.Size([64, 3, 32, 32])
print(f"첫 번째 배치 레이블 shape: {labels.shape}") # 예시: torch.Size([64])

# 모델의 입력 특성 개수 정의: CIFAR-10은 32x32 컬러(3채널) 이미지이므로 32*32*3 = 3072 입니다.
# 이 값은 모델 정의 시 `num_features`로 사용됩니다.
# NUM_FEATURES = 3 * 32 * 32 # 3072

# DNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 순차적인 레이어 정의
        self.layers = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(0.2),
            nn.LazyConv2d(64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.LazyConv2d(32, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(100) # CIFAR-100은 10개의 클래스
        )
        
    def forward(self, x):
        return self.layers(x)

# 모델 인스턴스 생성 시 올바른 num_features 값 사용
model = CNN().to(DEVICE)
print(f"모델 구조:\n{model}")

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)

# 학습 함수
def train(model, criterion, optimizer, loader):
    model.train() # 모델을 학습 모드로 설정
    epochs_loss = 0
    epochs_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad() # 이전 기울기 초기화
        
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward() # 역전파: 기울기 계산
        optimizer.step() # 옵티마이저: 가중치 업데이트
        
        y_pred = torch.argmax(hypothesis, 1) # 가장 높은 확률을 가진 클래스 예측
        acc = (y_pred == y_batch).float().mean() # 정확도 계산
        epochs_loss += loss.item()
        epochs_acc += acc

    return epochs_loss / len(loader), epochs_acc / len(loader)

# 평가 함수
def evaluate(model, criterion, loader):
    model.eval() # 모델을 평가 모드로 설정 (Dropout, BatchNorm 등이 비활성화됨)
    epochs_loss = 0
    epochs_acc = 0
    
    with torch.no_grad(): # 기울기 계산 비활성화 (메모리 절약, 속도 향상)
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean()
            
            epochs_loss += loss.item()
            epochs_acc += acc
    return epochs_loss / len(loader), epochs_acc / len(loader)

# 학습 루프
epochs = 100 # 에포크 수를 적절히 설정
for epoch in range(1, epochs + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    # 출력 포맷팅
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, '
          f'Val_loss: {val_loss:.4f}, Val_acc: {val_acc:.4f}')
    
# 최종 평가
final_loss, final_acc = evaluate(model, criterion, test_loader)
print(f'=====================================================')
print(f'Final Test Loss: {final_loss:.4f}, Final Test Acc: {final_acc:.4f}')

# Final Test Loss: 1.4843, Final Test Acc: 0.4707