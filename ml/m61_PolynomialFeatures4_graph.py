import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.family'] = 'Malgun Gothic'

#1. 데이터
np.random.seed(4132)
x = 2 * np.random.rand(100, 1) -1
print(np.max(x), np.min(x)) # 0.9825273683659088 -0.9961075806188975

y = 3 * x**2 + 2*x + 1 + np.random.randn(100, 1)    # y = 3x^2 + 2x + 1 + 노이즈

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_pf, y)

# 원래 데이터 그리기
plt.scatter(x, y, color='blue', label='Original Data')
plt.xlabel(x)
plt.ylabel(y)
plt.title('Polynomial Regression 예제')
# plt.show()

# 다항식 회귀 그래프 그리기
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_pf = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot_pf = model2.predict(x_test_pf)
plt.plot(x_test, y_plot, color='red', label='원래 데이터')
plt.plot(x_test, y_plot_pf, color='green', label='Polynomial Regression')

plt.legend()
plt.grid()
plt.show()