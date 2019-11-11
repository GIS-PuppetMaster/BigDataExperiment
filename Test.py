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
    x_test = np.expand_dims(x_test, -1)
    y_test_size = y_test.shape[0]
    model = keras.models.load_model('checkPoint.h5')
    predict = model.predict(x_test)
    eva = model.evaluate(x_test, y_test)
    print("mae:" + str(eva[1]))
    print("acc:" + str(eva[2]))
    y_pred_size = predict.shape[0]

    # 计算真实比率
    # 获取one-hot编码规则
    with open('OneHotEncoder.json', 'rb') as f:
        enc = pickle.load(f)
    y_test = enc.inverse_transform(y_test)
    predict = enc.inverse_transform(predict)

    y_test = Counter(np.squeeze(y_test))
    y_pred = Counter(np.squeeze(predict))
    for k, v in y_test.items():
        y_test[k] = v / y_test_size
    for k, v in y_pred.items():
        y_pred[k] = v / y_pred_size

    # 确保字典中包含所有分类
    class_list = ['Fail', 'Pass', 'Good', 'Excellent']
    for c in class_list:
        if c not in y_pred.keys():
            y_pred[c] = 0.0

    y_test = pd.DataFrame(y_test, index=[0]).values
    y_pred = pd.DataFrame(y_pred, index=[0]).values
    score = 1 / (1 + 10 * mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print("score:" + str(score))


test()
