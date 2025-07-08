import numpy as np

x = np.array([[15.3], [38.7], [60.3], [82.8], [99.9]])

x = np.rint(x).astype(int)

print(x)

# 반올림하고 int형으로 변환해줌
# ([ 15]
#  [ 39]
#  [ 60]
#  [ 83]
#  [100])