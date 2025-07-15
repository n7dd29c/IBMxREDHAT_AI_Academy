import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
print(data)
print(data.shape)                   # (1000,)
print(np.min(data), np.max(data))   # 0.01256210937609914 73.6076005800709

log_data = np.log1p(data)           # log1p : log0에 대한 오류를 해결
                                    # np.expm1(data) : 변환된 값을 되돌림
print(log_data)

plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Transformed Log')

plt.show()
