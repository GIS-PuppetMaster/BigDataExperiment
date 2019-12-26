import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.feature_selection import *

first = pd.read_csv(open('Data/初赛训练集.csv'))
second = pd.read_csv(open('Data/second_round_training_data.csv'))
data = pd.concat((first, second), axis=0, sort=False)
"""
# 剔除B类属性
for i in range(1, 11):
    data.pop('Attribute' + str(i))
"""
# 补全缺失值为0
# data = data.fillna(0)

"""
# 添加特征
# 对每列特征取arctan
for i in range(1,11):
    data['Enhance'+str(i)] = np.arctan(np.array(data['Parameter'+str(i)]))
"""

data.to_csv('Data/data.csv')

# 扩展标签dim为2
label = np.expand_dims(np.array(data['Quality_label']), 1)

# 对label进行One-Hot编码
# ['Fail','Pass','Good','Excellent']
enc = OneHotEncoder()
label = enc.fit_transform(label).toarray()

# 保存编码器
with open('OneHotEncoder.json', 'wb') as f:
    pickle.dump(enc, f)

# 去掉标签列，分离出x
data = data.drop(columns=['Quality_label'])
x = np.array(data)
x = x.astype(np.float64)

# 打乱x的样本顺序
# 导致训练集准确率迅速收敛，验证集准确率不变
# np.random.shuffle(x)

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

# 使用GBDT进行特征组合
gbdt_enc = OneHotEncoder()
gbdt = GradientBoostingClassifier(n_estimators=10)
gbdt.fit(x_train, enc.inverse_transform(y_train))
x_train_gbdt = np.array((gbdt_enc.fit_transform(gbdt.apply(x_train)[:, :, 0])).toarray())
x_test_gbdt = np.array((gbdt_enc.transform(gbdt.apply(x_test)[:, :, 0])).toarray())
# 对组合特征标准化
x_train_gbdt = StandardScaler().fit_transform(x_train_gbdt)
x_train_gbdt = StandardScaler().fit_transform(x_test_gbdt)
# 特征合并
x_train = np.concatenate((x_train, x_train_gbdt), axis=1)
x_test = np.concatenate((x_test, x_test_gbdt), axis=1)
# 保存编码器
with open('GBDT_enc.json', 'wb') as f:
    pickle.dump(gbdt_enc, f)
# 保存GBDT
with open('GBDT.json', 'wb') as f:
    pickle.dump(scaler, f)


# 保存训练集和测试集
pd.DataFrame(x_train).to_csv("Data/x_train.csv", index=False, float_format='%.16f')
pd.DataFrame(x_test).to_csv("Data/x_test.csv", index=False, float_format='%.16f')
pd.DataFrame(y_train).to_csv("Data/y_train.csv", index=False, float_format='%.16f')
pd.DataFrame(y_test).to_csv("Data/y_test.csv", index=False, float_format='%.16f')
