import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import *
import pickle

first = pd.read_csv(open('Data/初赛训练集.csv'))
second = pd.read_csv(open('Data/second_round_training_data.csv'))
data = pd.concat((first, second), axis=0, sort=False)
# 剔除B类属性
for i in range(1, 11):
    data.pop('Attribute'+str(i))

# 补全缺失值为0
data = data.fillna(0)
data.to_csv('Data/data.csv')

raw = np.array(data)
# 扩展dim为2
label = np.expand_dims(np.array(data['Quality_label']), 1)

# 对label进行One-Hot编码
# ['Fail','Pass','Good','Excellent']
enc = OneHotEncoder()
label = enc.fit_transform(label).toarray()

# 保存编码器
with open('OneHotEncoder.json', 'wb') as f:
    pickle.dump(enc, f)

# 分离x
x = raw[:, :-1]
x = x.astype(np.float64)

# 对x取log
x = np.log10(x)

# 对x标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 保存标准化器
with open('Scaler.json', 'wb') as f:
    pickle.dump(scaler, f)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.2, random_state=42)

# 保存训练集和测试集
pd.DataFrame(x_train).to_csv("Data/x_train.csv", index=False, float_format='%.16f')
pd.DataFrame(x_test).to_csv("Data/x_test.csv", index=False, float_format='%.16f')
pd.DataFrame(y_train).to_csv("Data/y_train.csv", index=False, float_format='%.16f')
pd.DataFrame(y_test).to_csv("Data/y_test.csv", index=False, float_format='%.16f')
