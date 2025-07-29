from keras.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import torch
import numpy as np

###################### 랜덤 고정 ######################
SEED = 1
random.seed(SEED)               # python 랜덤 고정
np.random.seed(SEED)            # numpy 랜덤 고정
torch.manual_seed(SEED)         # torch 랜덤 고정
torch.cuda.manual_seed(SEED)    # torch, cuda 시드 고정
#######################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
