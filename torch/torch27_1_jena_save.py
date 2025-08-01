import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
# matplotlib.pyplot은 현재 코드에서는 사용되지 않아 주석 처리했습니다.
# import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler # MinMaxScaler 추가

SEED = 3112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = './_data/kaggle/jena/'
jena_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(f"Original jena_csv shape: {jena_csv.shape}")     # (420551, 14)

# 1. 데이터 전처리: 결측치 확인 및 처리 (필요시)
# print(jena_csv.info()) # 데이터 타입 및 결측치 확인
# print(jena_csv.head())

# Time 컬럼을 datetime으로 변환 (필요하다면)
# jena_csv['Date Time'] = pd.to_datetime(jena_csv['Date Time'])

# 결측치 확인 및 제거/대체 (예시: NaN이 있다면 제거)
# jena_csv.dropna(inplace=True)

# 2. MinMaxScaler 적용 (전체 데이터에 스케일링)
# Target 컬럼은 'T (degC)'로 가정하고 마지막 컬럼으로 옮겨 학습에 사용합니다.
# 다른 컬럼들도 스케일링하는 것이 일반적입니다.
features_to_scale = jena_csv.columns # 'Date Time' 제외
scaler = MinMaxScaler()
jena_scaled = scaler.fit_transform(jena_csv[features_to_scale])
jena_scaled_df = pd.DataFrame(jena_scaled, columns=features_to_scale)

# 'wd (deg)'를 마지막 컬럼으로 재배열 (예시)
# 예측하고자 하는 대상이 'wd (deg)'라고 가정합니다.
# 만약 다른 컬럼을 예측하고 싶다면 해당 컬럼을 선택합니다.
target_column_name = 'wd (deg)' # 또는 예측하고자 하는 다른 컬럼 이름

# 입력 X와 타겟 Y 분리
# Y는 예측하고자 하는 'wd (deg)'로 가정
# X는 'Date Time'과 'wd (deg)'를 제외한 나머지 모든 컬럼
X_data = jena_scaled_df.drop(columns=[target_column_name]).values.astype(np.float32)
y_data = jena_scaled_df[target_column_name].values.astype(np.float32)

print(f"X_data shape after scaling: {X_data.shape}")
print(f"y_data shape after scaling: {y_data.shape}")


class CustomData(Dataset):
    def __init__(self, X, Y, timesteps, strides):
        self.X = X # 스케일링된 피처 데이터
        self.Y = Y # 스케일링된 타겟 데이터
        self.timesteps = timesteps # 입력 시퀀스 길이
        self.strides = strides     # 시퀀스 생성 시 건너뛸 간격

        # 시퀀스 생성 (데이터셋 초기화 시점에 모두 생성)
        self.sequences_x = []
        self.sequences_y = []

        self.target_offset = 144 # 1440분 / 10분 = 144 (24시간)
        
        max_start_idx = len(self.Y) - self.timesteps - self.target_offset 
        
        for i in range(0, max_start_idx + 1, self.strides):
            x_seq = self.X[i : i + self.timesteps]
            y_val = self.Y[i + self.timesteps + self.target_offset -1] # -1은 인덱스 보정
            
            self.sequences_x.append(x_seq)
            self.sequences_y.append(y_val)
            
        # 리스트를 NumPy 배열로 변환
        self.sequences_x = np.array(self.sequences_x, dtype=np.float32)
        self.sequences_y = np.array(self.sequences_y, dtype=np.float32)

        print(f"Generated X sequences shape: {self.sequences_x.shape}")
        print(f"Generated Y sequences shape: {self.sequences_y.shape}")

    def __len__(self):
        return len(self.sequences_x) # 생성된 시퀀스의 개수

    def __getitem__(self, idx):
        # 이미 __init__에서 모든 시퀀스를 생성했으므로, 인덱스에 따라 가져오기만 합니다.
        return self.sequences_x[idx], self.sequences_y[idx]

# 데이터셋 초기화 시 df 대신 X_data, y_data 전달
custom_data = CustomData(X=X_data, Y=y_data, timesteps=144, strides=144) 
train_loader = DataLoader(custom_data, batch_size=32, shuffle=False) # shuffle=True 권장하지만, 데이터 저장 목적이면 False

# DataLoader를 통해 데이터 확인 (하나의 배치만 확인)
for batch_idx, (x, y) in enumerate(train_loader):
    print(f'\n================== {batch_idx}번 batch ==================')
    print(f'x batch shape: {x.shape}') # (batch_size, timesteps, features)
    print(f'y batch shape: {y.shape}') # (batch_size,)
    break # 첫 번째 배치만 확인하고 종료

# 전체 데이터셋을 numpy 배열로 저장하려면, DataLoader를 다시 순회해야 합니다.
# DataLoader는 데이터를 batch_size 단위로 끊어주므로,
# 모든 배치를 모아서 하나의 큰 numpy 배열로 concatenate 해야 합니다.
all_x_data = []
all_y_data = []

# tqdm을 사용하여 진행 상황을 시각화할 수 있습니다.
# from tqdm import tqdm
# for x_batch, y_batch in tqdm(train_loader, desc="Collecting all data"):
for x_batch, y_batch in train_loader:
    all_x_data.append(x_batch.cpu().numpy()) # GPU에 있다면 CPU로 옮겨서 numpy 변환
    all_y_data.append(y_batch.cpu().numpy())

# 리스트에 있는 배치들을 하나의 큰 배열로 합칩니다.
# x는 (N, timesteps, features), y는 (N,) 형태가 됩니다.
final_x_array = np.concatenate(all_x_data, axis=0)
final_y_array = np.concatenate(all_y_data, axis=0)

print(f"\nFinal x array shape to save: {final_x_array.shape}")
print(f"Final y array shape to save: {final_y_array.shape}")

np.save('./_data/torch/jena/jena_x_data.npy', final_x_array)
np.save('./_data/torch/jena/jena_y_data.npy', final_y_array)
print('저장 완료')
