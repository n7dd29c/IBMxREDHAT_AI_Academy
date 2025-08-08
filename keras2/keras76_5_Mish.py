import numpy as np
import matplotlib.pyplot as plt

# keras76_5_Mish.py
x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1+np.exp(x)))

y = mish(x)
