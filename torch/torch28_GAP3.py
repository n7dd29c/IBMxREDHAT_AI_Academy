import torch
import torch.nn as nn
import torch.nn.functional as Fn

x = torch.randn(1, 64, 10, 10)

#################### 1. AdaptiveAvgPool2D ####################
# gap = nn.AdaptiveAvgPool2d((1,1))
# x = gap(x)

#################### 2. AdaptiveAvgPool2D 한줄로 ####################
# x = nn.AdaptiveAvgPool2d((1,1))(x)  # 위와 같음

#################### 3. Fn.adaptive_avg_pool2d ####################
x = Fn.adaptive_avg_pool2d(x, (1,1))

print(x.shape)  # torch.Size([1, 64, 1, 1])

x = x.view(x.size(0), -1)
print(x.shape)  # torch.Size([1, 64])