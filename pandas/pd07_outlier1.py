import numpy as np
aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])

def outlier(data):
    quartile_1, q2, quartile_3 =  np.percentile(data, [25,50,75])
    print('1사분위 :', quartile_1)
    print('2사분위 :', q2)
    print('3사분위 :', quartile_3)
    
    iqr = quartile_3 - quartile_1
    print('IQR :', iqr)
    
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound)),\
        iqr, lower_bound, upper_bound

outlier_loc, iqr, low, upp = outlier(aaa)
print('이상치의 위치 :', outlier_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(upp, color='pink', label='upper bound')
plt.axhline(low, color='pink', label='lower bound')
plt.legend()
plt.show()

# boxplot의 네모난 범위는 1분위부터 3분위까지
# 네모난 범위 바로 위쪽에 있는 선은 이상치를 제외한 최대/최소값
# axhline으로 upper, lower bound를 표시