import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

alpha = 1.07123719821389
lmbda = 1.05070187983129

def selu(x, alpha, lmbda):
    return lmbda * ((x<0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

# elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

y = selu(x, 1.67, 1.05)
