import numpy as np
import pandas as pd

arr = np.arange(6).reshape(3,2)
df = pd.DataFrame(arr, columns=['a','b'])
print(df)

dataSet = pd.Series
