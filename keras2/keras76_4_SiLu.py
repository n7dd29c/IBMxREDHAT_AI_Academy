import numpy as np
import matplotlib.pyplot as plt

# keras76_4_SiLu.py
x = np.arange(-5, 5, 0.1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

def silu(x):
    return x * (1 / 1+np.exp(-x))

x = lambda x : x * (1 / 1+np.exp(-x))

y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()
