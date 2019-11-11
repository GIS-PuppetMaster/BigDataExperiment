from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import pickle

x_train = np.array(pd.read_csv('Data/x_train.csv'))
# x_train = np.expand_dims(x_train, -1)
y_train = pd.read_csv('Data/y_train.csv')
x_test = np.array(pd.read_csv('Data/x_test.csv'))
# x_test = np.expand_dims(x_test, -1)
y_test = pd.read_csv('Data/y_test.csv')

# 计算真实比率
# 获取one-hot编码规则
with open('OneHotEncoder.json', 'rb') as f:
    enc = pickle.load(f)
y_train = enc.inverse_transform(y_train).flatten()
y_test = enc.inverse_transform(y_test).flatten()
"""
svc = SVC(kernel='rbf', gamma='auto')
svc.fit(x_train, y_train)
metric = cross_val_score(svc, x_test, y_test, cv=5)
print("SVM_Score:" + str(metric))
"""
xgb = XGBClassifier().fit(x_train, y_train)
score = xgb.score(x_test, y_test)
print("XGBoost_Score:" + str(score))
