import warnings
import tensorflow as tf
import sklearn
from keras.utils import plot_model
from sklearn.preprocessing import *
import keras
from keras.layers import *
from keras.models import *
import numpy as np
import pandas as pd
from MyKerasTool import *
import matplotlib.pyplot as plt
import os
import shutil
import pickle
from Test import test
print("加载CheckPoint")
model = load_model('checkPointB.h5')
with open('checkPoint_LRB.json', 'rb') as f:
    temp_lr = pickle.load(f)
order = input("是否使用上次学习率:" + str(temp_lr) + "？(y/n)")
if order == 'y':
    lr = temp_lr
opt = keras.optimizers.Adam(lr)
model.compile(opt, loss=keras.losses.mean_squared_error, metrics=['mae', 'mse'])
model.summary()
print("加载学习率:" + str(lr))