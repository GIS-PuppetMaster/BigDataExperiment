import keras.models
from keras.metrics import *
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
import pickle


def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x_test = pd.read_csv('Data/x_test.csv')
    y_test = np.array(pd.read_csv('Data/y_test.csv'))
    x_test = np.array(x_test)
    # 剔除Attr1-3
    # x_test = np.concatenate((x_test[:, 0:10], x_test[:, 13:]), axis=1)

    x_test = np.expand_dims(x_test, -1)
    y_test_size = y_test.shape[0]
    # 预测Attr并使用预测值替换ground-truth
    modelB = keras.models.load_model('checkPointB.h5')
    x = x_test[:, 0:10]
    x_testB = modelB.predict(x)

    # 删除原先标记的所有数据
    x_test = np.delete(x_test, [i for i in range(10, 20)], axis=1)
    # 若剔除Attr1-3，则使用这行代码替代上面的
    # x_test = np.delete(x_test, [i for i in range(10, 17)], axis=1)

    # 插入预测的B类特征
    for i in range(x_testB.shape[1]):
        x_test = np.insert(arr=x_test, obj=10 + i, values=x_testB[:, i].reshape(-1, 1), axis=1)
    model = keras.models.load_model('checkPoint.h5')
    predict = model.predict(x_test)
    eva = model.evaluate(x_test, y_test)
    print("mae:" + str(eva[1]))
    print("acc:" + str(eva[2]))

    # 计算真实比率
    # 获取one-hot编码规则
    with open('OneHotEncoder.json', 'rb') as f:
        enc = pickle.load(f)
    y_test = enc.inverse_transform(y_test)
    predict = enc.inverse_transform(predict)

    ans_test = None
    ans_pred = None
    class_list = ['Fail', 'Pass', 'Good', 'Excellent']

    # 将测试数据切分为组，每组大小50
    for start in range(0, y_test_size, 50):
        group_test = y_test[start:start + 50, :]
        group_pred = predict[start:start + 50, :]
        group_test_size = group_test.shape[0]
        group_pred_size = group_pred.shape[0]
        group_test = Counter(np.squeeze(group_test))
        group_pred = Counter(np.squeeze(group_pred))
        for k, v in group_test.items():
            group_test[k] = v / group_test_size
        for k, v in group_pred.items():
            group_pred[k] = v / group_pred_size
        # 确保字典中包含所有分类
        for c in class_list:
            if c not in group_pred.keys():
                group_pred[c] = 0.0
        # 构造ratio行
        label_test = np.array(
            [group_test['Excellent'], group_test['Good'], group_test['Pass'], group_test['Fail']]).reshape(1, 4)
        label_pred = np.array(
            [group_pred['Excellent'], group_pred['Good'], group_pred['Pass'], group_pred['Fail']]).reshape(1, 4)
        # 构造ratio列表
        if ans_test is None:
            ans_test = label_test
        else:
            ans_test = np.concatenate((ans_test, label_test), axis=0)
        if ans_pred is None:
            ans_pred = label_pred
        else:
            ans_pred = np.concatenate((ans_pred, label_pred), axis=0)
    score = 1 / (1 + 10 * mean_absolute_error(y_true=ans_test, y_pred=ans_pred))
    print("score:" + str(score))

# test()
