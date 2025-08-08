import numpy as np
import matplotlib.pyplot as plt

# keras76_6_elu.py

x = np.arange(-5, 5, 0.1)

def elu(x, alpha):
    return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

y = elu(x, 1)
