import pandas as pd

df = pd.DataFrame({'X1' : [1,2,3,4,5],
                   'X2' : [5,4,3,2,1],
                   'X3' : [-1000, 50, 7, 100, -1100],
                   'Y' : [2,3,4,5,6],
                   })
print(df,'\n')

correlations = df.corr()
print(correlations)
