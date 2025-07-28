import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from sklearn.preprocessing import StandardScaler

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])
# ToTensor : torch.tensor로 바꾸기 + MinMaxScaler 적용
# Z_score Nomalization 정규화의 표준화?
# (x - 평균) / 표준편차
# (x - 0.5) / 0.5          위 식처럼 해야하는데 통상 평균 0.5, 표편 0.5로 계산하면
# -1 ~ 1 사이의 범위가 나오니 이미지 전처리에서는 0.5 0.5로 처리한다

path = './Study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(len(train_dataset))               # 60000
print(train_dataset[0][0])
print(train_dataset[0][1])              # 5

img_tensor, label = train_dataset[0]    # tuple 형태의 데이터는 이렇게 나눌 수 있다
print(label)                            # 5
print(img_tensor.shape)                 # torch.Size([1, 56, 56])
print(img_tensor.min(),                 # tensor(0.)
      img_tensor.max())                 # tensor(0.9922)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
print(len(train_loader))                # 1875

class CNN(nn.Module):
    def __init__(self, num_features):
        # super().__init__()
        super(CNN, self).__init__() # 위에랑 같은 말
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=(3,3), stride=1),  # (1,56,56) -> (128,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                  # (n,128,27,27)
        )
    
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), stride=1),            # (n,64,25,25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)                   # (n,64,12,12)
            
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),             # (n,32,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                  # (n,32,5,5)
        )
        
        self.flatten = nn.Flatten()
        
        self.hidden_layer4 = nn.Sequential(                             # Flatten에서 받아야 함
            nn.LazyLinear(16),       
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(16, 10)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = CNN(1).to(DEVICE)   # torch에서는 channel만 input으로 넣어주면 된다, 나머지는 알아서

# model.summary()   # error

print(model)
# CNN(
#   (hidden_layer1): Sequential(
#     (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (hidden_layer2): Sequential(
#     (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (hidden_layer3): Sequential(
#     (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (hidden_layer4): Sequential(
#     (0): LazyLinear(in_features=0, out_features=16, bias=True)
#     (1): ReLU()
#   )
#   (output_layer): Linear(in_features=16, out_features=10, bias=True)
#   (drop): Dropout(p=0.2, inplace=False)
# )

from torchsummary import summary
summary(model, (1, 56, 56))
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1          [-1, 128, 54, 54]           1,280
#               ReLU-2          [-1, 128, 54, 54]               0
#          MaxPool2d-3          [-1, 128, 27, 27]               0
#             Conv2d-4           [-1, 64, 25, 25]          73,792
#               ReLU-5           [-1, 64, 25, 25]               0
#          MaxPool2d-6           [-1, 64, 12, 12]               0
#             Conv2d-7           [-1, 32, 10, 10]          18,464
#               ReLU-8           [-1, 32, 10, 10]               0
#          MaxPool2d-9             [-1, 32, 5, 5]               0
#           Flatten-10                  [-1, 800]               0
#            Linear-11                   [-1, 16]          12,816
#              ReLU-12                   [-1, 16]               0
#            Linear-13                   [-1, 10]             170
# ================================================================
# Total params: 106,522
# Trainable params: 106,522
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 7.15
# Params size (MB): 0.41
# Estimated Total Size (MB): 7.57
# ----------------------------------------------------------------

from torchinfo import summary
summary(model, (32, 1, 56, 56))
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# CNN                                      [32, 10]                  --
# ├─Sequential: 1-1                        [32, 128, 27, 27]         --
# │    └─Conv2d: 2-1                       [32, 128, 54, 54]         1,280
# │    └─ReLU: 2-2                         [32, 128, 54, 54]         --
# │    └─MaxPool2d: 2-3                    [32, 128, 27, 27]         --
# ├─Sequential: 1-2                        [32, 64, 12, 12]          --
# │    └─Conv2d: 2-4                       [32, 64, 25, 25]          73,792
# │    └─ReLU: 2-5                         [32, 64, 25, 25]          --
# │    └─MaxPool2d: 2-6                    [32, 64, 12, 12]          --
# ├─Sequential: 1-3                        [32, 32, 5, 5]            --
# │    └─Conv2d: 2-7                       [32, 32, 10, 10]          18,464
# │    └─ReLU: 2-8                         [32, 32, 10, 10]          --
# │    └─MaxPool2d: 2-9                    [32, 32, 5, 5]            --
# ├─Flatten: 1-4                           [32, 800]                 --
# ├─Sequential: 1-5                        [32, 16]                  --
# │    └─Linear: 2-10                      [32, 16]                  12,816
# │    └─ReLU: 2-11                        [32, 16]                  --
# ├─Linear: 1-6                            [32, 10]                  170
# ==========================================================================================
# Total params: 106,522
# Trainable params: 106,522
# Non-trainable params: 0
# Total mult-adds (G): 1.65
# ==========================================================================================
# Input size (MB): 0.40
# Forward/backward pass size (MB): 106.62
# Params size (MB): 0.43
# Estimated Total Size (MB): 107.44
# ==========================================================================================