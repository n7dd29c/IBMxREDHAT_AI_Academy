from sklearn.datasets import fetch_california_housing
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seed=3112
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target

df.boxplot()
plt.show()