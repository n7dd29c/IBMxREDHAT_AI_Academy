import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
print(pd.value_counts(y_test))

aaa = 1
print(y_train[aaa])

import matplotlib.pyplot as plt
plt.imshow(x_train[aaa], 'gray')
plt.show()