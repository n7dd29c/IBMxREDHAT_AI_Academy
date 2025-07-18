import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [-100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

print(aaa, aaa.shape)   # (13, 2)

outlier_indices = []
iqrs = []
lower_bounds = []
upper_bounds = []

def outlier(data):
    # 데이터의 각 열(feature)에 대해 반복   
    for i in range(data.shape[1]):
        col_data = data[:, i] # 현재 열의 데이터

        q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
        print(f'\n--- Column {i} ---')
        print('1사분위 :', q1)
        print('2사분위 :', q2)
        print('3사분위 :', q3)

        iqr = q3 - q1
        print('IQR :', iqr)

        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        print(f'Lower Bound : {lower_bound}')
        print(f'Upper Bound : {upper_bound}')

        # 이상치 위치 저장 (원본 배열의 인덱스 기준)
        outlier_col_indices = np.where((col_data > upper_bound) | (col_data < lower_bound))
            
        # (행 인덱스, 열 인덱스) 형태로 저장하기 위해 변환
        for r_idx in outlier_col_indices[0]:
            outlier_indices.append((r_idx, i)) # (행 인덱스, 열 인덱스) 튜플 추가

            iqrs.append(iqr)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
                
    return outlier_indices, iqrs, lower_bounds, upper_bounds # 반환 값 변경

# outlier 함수 호출
outlier_locs, iqrs, lows, upps = outlier(aaa)

print('\n이상치의 위치 (행, 열) :', outlier_locs)

print(f'\n각 열별 IQR: {iqrs}')
print(f'각 열별 Lower Bounds : {lows}')
print(f'각 열별 Upper Bounds : {upps}')

# 이상치의 값 출력
print('\n--- 이상치의 실제 값 ---')
if outlier_locs: # 이상치가 존재하는 경우에만 출력
    for r_idx, c_idx in outlier_locs:
        print(f'위치 ({r_idx}, {c_idx}): {aaa[r_idx, c_idx]}')
else:
    print('이상치가 없습니다.')