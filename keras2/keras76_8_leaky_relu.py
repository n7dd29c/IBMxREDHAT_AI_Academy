import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

alpha = 0.01
def leaky_relu(x):
    return np.maximum(alpha*x, x)

# leaky_relu = lambda x :np.where(x>0, x, alpha*x)  # 위랑 같음

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()