import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4, 3)
print(x)

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
x_pf = pf.fit_transform(x)
print(x_pf)

# interaction_only = True <- 단순곱
# [[  0.   1.   2.   0.   0.   2.]
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]