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
from Test import test

if os.path.isdir('./tensor_board_logs'):
    shutil.rmtree('./tensor_board_logs')

x_input = Input((10, 1))

x1 = Conv1D(filters=4, kernel_size=3, strides=1, dilation_rate=2, padding='same',
            kernel_regularizer=keras.regularizers.l2(0.01))(x_input)
x1 = BatchNormalization(epsilon=1e-4)(x1)
x1 = Dropout(0.2)(x1)
x1 = ReLU()(x1)
x1 = Conv1D(filters=4, kernel_size=2, strides=1, dilation_rate=3, padding='same',
            kernel_regularizer=keras.regularizers.l2(0.01))(x1)
x1 = BatchNormalization(epsilon=1e-4)(x1)
x1 = Dropout(0.2)(x1)
x1 = ReLU()(x1)

x2 = Conv1D(filters=4, kernel_size=5, strides=1, kernel_regularizer=keras.regularizers.l2(0.001))(x_input)
x2 = BatchNormalization(epsilon=1e-4)(x2)
x2 = Dropout(0.2)(x2)
x2 = ReLU()(x2)
x2 = Conv1D(filters=8, kernel_size=3, strides=2, kernel_regularizer=keras.regularizers.l2(0.001))(x2)
x2 = BatchNormalization(epsilon=1e-4)(x2)
x2 = Dropout(0.2)(x2)
x2 = ReLU()(x2)

x3 = Conv1D(filters=8, kernel_size=5, strides=2, kernel_regularizer=keras.regularizers.l2(0.001))(x_input)
x3 = BatchNormalization(epsilon=1e-4)(x3)
x3 = Dropout(0.2)(x3)
x3 = ReLU()(x3)
x3 = Conv1D(filters=8, kernel_size=3, strides=1, kernel_regularizer=keras.regularizers.l2(0.001))(x3)
x3 = BatchNormalization(epsilon=1e-4)(x3)
x3 = Dropout(0.2)(x3)
x3 = ReLU()(x3)

x4 = Conv1D_conv_block(x_input, filters=(16, 8, 16), block_name='stage1_conv-', data_format='channels_last')
for i in range(2):
    x4 = Conv1D_identity_block(x4, filters=(16, 8, 16),
                               block_name='stage1_identity_' + str(i) + '-', data_format='channels_last')
x4 = Conv1D_conv_block(x4, filters=(8, 4, 8), strides=2, block_name='stage2_conv-', data_format='channels_last')
for i in range(3):
    x4 = Conv1D_identity_block(x4, filters=(8, 4, 8),
                               block_name='stage2_identity_' + str(i) + '-', data_format='channels_last')

x5 = Dense_BN(x_input, 32)
x5 = Dense_BN(x5, 128)
x5 = Dense_BN(x5, 32)
x5 = Dense_BN(x5, 16)


x1 = Flatten()(x1)
x2 = Flatten()(x2)
x3 = Flatten()(x3)
x4 = Flatten()(x4)
x5 = Flatten()(x5)

x = Concatenate()([x1, x2, x3, x4, x5])
x = Dense_BN(x, 32)
x = Dense(4, activation='softmax')(x)

model = Model(inputs=[x_input], outputs=[x])
plot_model(model, to_file='model.png', show_shapes=True)
model.compile(keras.optimizers.Adam(0.01), loss=keras.losses.categorical_crossentropy, metrics=['mae', 'acc'])

x_train = np.array(pd.read_csv('Data/x_train.csv'))
x_train = np.expand_dims(x_train, -1)
y_train = pd.read_csv('Data/y_train.csv')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=0.001)
# early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=300, restore_best_weights=True)
check_point = keras.callbacks.ModelCheckpoint(filepath='./checkPoint.h5', monitor='val_acc', save_best_only=True,
                                              period=20)
tensor_board = keras.callbacks.TensorBoard(log_dir='./tensor_board_logs', write_grads=True, write_graph=True,
                                           write_images=True)
history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1,
                    callbacks=[reduce_lr, check_point, tensor_board], verbose=2,
                    batch_size=512, shuffle=True)
test()
"""
model.save('model.h5')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE(Metric)')
plt.plot(hist['epoch'], hist['mean_absolute_error'], label='train_MAE(Metric)')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='val_MAE(Metric)')
plt.legend()
plt.savefig('Loss.png')
plt.show()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(hist['epoch'], hist['acc'], label='train_Acc')
plt.plot(hist['epoch'], hist['val_acc'], label='val_Acc')
plt.legend()
plt.savefig('Acc.png')
plt.show()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss(CrossEntropy)')
plt.plot(hist['epoch'], hist['loss'], label='train_loss')
plt.plot(hist['epoch'], hist['val_loss'], label='val_loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()
"""
