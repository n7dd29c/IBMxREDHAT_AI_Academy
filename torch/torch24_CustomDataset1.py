import torch
from torch.utils.data import DataLoader, Dataset

#1. 커스텀 데이터셋 만들기: TensorDataset에 비해 제약이 없어 전처리가 자유롭다
class MyData(Dataset):
    def __init__(self):
        self.x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.y = [0,1,0,1,0]
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])
    
#2. 인스턴스 생성
dataset = MyData()

#3. DataLoader에 집어넣기
loader = DataLoader(dataset, batch_size=2, shuffle=True)

#4. 출력
for batch_idx, (xb, yb) in enumerate(loader):   # DataLoader는 iterator형태
    print(f'\n================== {batch_idx}번 batch ==================')
    print(f'x : {xb}\ny : {yb}')