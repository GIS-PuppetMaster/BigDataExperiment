import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import stats
from scipy.stats import norm

data = pd.read_csv('Data/data.csv')
# 查看数据整体信息
des = data.describe()
des.to_csv('Data/DataDescribe.csv')
# 查看数据分布情况
data_nd = np.array(data)
p = plt.figure(figsize=(20,10))
for i in range(1, 21):
    p.add_subplot(2,10,i)
    name = data.columns.values[i]
    plt.title(name)
    y = pd.DataFrame((data_nd[:, i].astype(np.float64)))
    sns.distplot(y, kde=True, fit=st.norm)
plt.show()
