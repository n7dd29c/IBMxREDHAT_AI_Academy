import numpy as np
import matplotlib.pyplot as plt

#keras76_3_relu.py
x = np.arange(-5, 5, 0.1)

def relu(x):
    return np.maximum(0, x)

x = lambda x : np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

