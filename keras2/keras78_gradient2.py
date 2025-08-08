import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
gradient = lambda x : 2*x - 4

x = -10.0   # 초기값
epoch = 50
leaning_rate = 0.1

print('epoch \t x \t F(x)')
print('{:02d}\t {:0.5f}\t {:6.5f}\t'.format(0, x, f(x)))

for i in range(epoch ):
    x = x - leaning_rate * gradient(x)
    print('{:02d}\t {:0.5f}\t {:6.5f}\t'.format(i+1, x, f(x)))