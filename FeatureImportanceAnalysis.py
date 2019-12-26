import pickle
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import *
import numpy as np
import pandas as pd
from math import *
from xgboost import XGBClassifier
from pymining import itemmining, assocrules


def print_col_index(x_new, mod):
    for i in range(1, 11):
        if np.all(x_new == dataset[:, i].reshape(-1, 1)):
            print(mod + "特征选择：param" + str(i))


# 载入数据集
dataset = np.array(pd.read_csv(open('Data/data.csv')))
x = dataset[:, 1:11]
y = dataset[:, -1]

# 单变量方差特征选择
selector = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=10)
x_new = selector.fit_transform(x, y)
# 判断选择出的是第几列
print_col_index(x_new, "单变量方差分析")
print("")

# 单变量卡方特征选择
selector = GenericUnivariateSelect(score_func=chi2, mode='percentile', param=10)
x_new = selector.fit_transform(x, y)
# 判断选择出的是第几列
print_col_index(x_new, "单变量卡方")
print("")

# 循环特征选择
selector = RFECV(XGBClassifier(), scoring=sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score), n_jobs=-1)
x_new = selector.fit_transform(x, y)
for i in range(x_new.shape[1]):
    print_col_index(x_new[:, i].reshape(-1, 1), "XGBoost-AccScoring循环")
print("")

# 单变量关联分析
for i in range(1, x.shape[1] + 1):
    # 提取元组
    # data = dataset[:, [i, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]]
    data = dataset[:, [i, -1]]
    # 按特征大小排序
    data = np.array(
        # pd.DataFrame(data, columns=['param', 'att1','att2','att3','att4','att5','att6','att7','att8','att9','att10','label']).sort_values(axis=0, by='param', ascending=True, inplace=False))
        pd.DataFrame(data,
                     columns=['param', 'label']).sort_values(axis=0, by='param', ascending=True, inplace=False))

    # 使用等频离散将feature离散化
    # 区间数目
    k = 20
    # 区间内记录数目
    n = ceil(x.shape[0] / k)
    # 区间划分[front,last)
    cells = []
    front = 0
    last = front + n + 1
    while last <= data.shape[0]:
        cells.append(data[front:last, :])
        front = last
        last = front + n + 1
    # 离散化区间
    des_cell = cells.copy()
    # 区间编号j
    for j in range(len(des_cell)):
        for k in range(des_cell[j].shape[0]):
            des_cell[j][k][0] = j

    # 使用APRIORI算法进行关联分析
    # 将离散化区间转为元组
    tup_list = []
    for j in range(len(des_cell)):
        for k in range(des_cell[j].shape[0]):
            tup = (des_cell[j][k][0], des_cell[j][k][1])
            tup_list.append(tup)
    transactions = tuple(tup_list)
    relim_input = itemmining.get_relim_input(transactions)
    item_sets = itemmining.relim(relim_input, min_support=1)
    rules = assocrules.mine_assoc_rules(item_sets, min_support=1, min_confidence=0.1)
    print(rules)
