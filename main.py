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


class CheckPoint_Save_LR(keras.callbacks.Callback):
    """Save the model after every epoch.

        `filepath` can contain named formatting options,
        which will be filled with the values of `epoch` and
        keys in `logs` (passed in `on_epoch_end`).

        For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
        then the model checkpoints will be saved with the epoch number and
        the validation loss in the filename.

        # Arguments
            filepath: string, path to save the model file.
            monitor: quantity to monitor.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.
            save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).
            mode: one of {auto, min, max}.
                If `save_best_only=True`, the decision
                to overwrite the current save file is made
                based on either the maximization or the
                minimization of the monitored quantity. For `val_acc`,
                this should be `max`, for `val_loss` this should
                be `min`, etc. In `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.
            period: Interval (number of epochs) between checkpoints.
        """

    def __init__(self, filepath, optimizer, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CheckPoint_Save_LR, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.optimizer = optimizer
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        with open('checkPoint_LR.json', 'wb') as f:
                            session = keras.backend.get_session()
                            lr = opt.lr.eval(session=session)
                            pickle.dump(lr, f)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


lr = 0.05

if os.path.isdir('./tensor_board_logs'):
    shutil.rmtree('./tensor_board_logs')
opt = None
if not os.path.exists('./checkPoint.h5'):
    print("未找到CheckPoint，重新初始化模型")
    x_input = Input((93, 1))

    x1 = Conv1D(filters=4, kernel_size=3, strides=1, dilation_rate=2, padding='same')(x_input)
    x1 = BatchNormalization(epsilon=1e-4)(x1)
    x1 = Dropout(0.5)(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(filters=4, kernel_size=2, strides=1, dilation_rate=3, padding='same',
                kernel_regularizer=keras.regularizers.l2(0.01))(x1)
    x1 = BatchNormalization(epsilon=1e-4)(x1)
    x1 = Dropout(0.5)(x1)
    x1 = ReLU()(x1)

    x2 = Conv1D(filters=4, kernel_size=5, strides=1)(x_input)
    x2 = BatchNormalization(epsilon=1e-4)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = ReLU()(x2)
    x2 = Conv1D(filters=8, kernel_size=3, strides=2)(x2)
    x2 = BatchNormalization(epsilon=1e-4)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = ReLU()(x2)

    x3 = Conv1D(filters=8, kernel_size=5, strides=2)(x_input)
    x3 = BatchNormalization(epsilon=1e-4)(x3)
    x3 = Dropout(0.5)(x3)
    x3 = ReLU()(x3)
    x3 = Conv1D(filters=8, kernel_size=3, strides=1)(x3)
    x3 = BatchNormalization(epsilon=1e-4)(x3)
    x3 = Dropout(0.5)(x3)
    x3 = ReLU()(x3)

    x4 = Conv1D_conv_block(x_input, filters=(16, 8, 16), block_name='stage1_conv-', data_format='channels_last')
    for i in range(2):
        x4 = Conv1D_identity_block(x4, filters=(16, 8, 16),
                                   block_name='stage1_identity_' + str(i) + '-', data_format='channels_last')
    x4 = Conv1D_conv_block(x4, filters=(8, 4, 8), strides=2, block_name='stage2_conv-', data_format='channels_last')
    for i in range(3):
        x4 = Conv1D_identity_block(x4, filters=(8, 4, 8),
                                   block_name='stage2_identity_' + str(i) + '-', data_format='channels_last')

    x5 = Dense_BN(x_input, 32, activation='relu')
    x5 = Dense_BN(x5, 128, activation='relu')
    x5 = Dense_BN(x5, 32, activation='relu')
    x5 = Dense_BN(x5, 16, activation='relu')

    x6 = CuDNNLSTM(10, return_sequences=True)(x_input)
    x6 = BatchNormalization()(x6)
    x6 = Dropout(0.5)(x6)
    x6 = Activation('tanh')(x6)
    x6 = CuDNNLSTM(10, return_sequences=False)(x6)
    x6 = BatchNormalization()(x6)
    x6 = Dropout(0.5)(x6)
    x6 = Activation('tanh')(x6)

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)
    x4 = Flatten()(x4)
    x5 = Flatten()(x5)
    # x6 = Flatten()(x6)

    x = Concatenate()([x1, x2, x3, x4, x5, x6])
    x = Dense_BN(x, 64)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=[x_input], outputs=[x])
    plot_model(model, to_file='model.png', show_shapes=True)
    opt = keras.optimizers.Adam(lr)
    model.compile(opt, loss=keras.losses.categorical_crossentropy, metrics=['mae', 'acc'])
else:
    print("加载CheckPoint")
    model = load_model('checkPoint.h5')
    with open('checkPoint_LR.json', 'rb') as f:
        temp_lr = pickle.load(f)
    order = input("是否使用上次学习率:" + str(temp_lr) + "？(y/n)")
    if order == 'y':
        lr = temp_lr
    opt = keras.optimizers.Adam(lr)
    model.compile(opt, loss=keras.losses.categorical_crossentropy, metrics=['mae', 'acc'])
    model.summary()
    print("加载学习率:" + str(lr))

x_train = np.array(pd.read_csv('Data/x_train.csv'))
x_train = np.concatenate((x_train[:, 0:10], x_train[:, 13:]), axis=1)
x_train = np.expand_dims(x_train, -1)
y_train = pd.read_csv('Data/y_train.csv')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=0.0001)
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=400, restore_best_weights=True)
check_point = CheckPoint_Save_LR(filepath='./checkPoint.h5', monitor='val_acc', optimizer=opt, save_best_only=True,
                                 verbose=1)
tensor_board = keras.callbacks.TensorBoard(log_dir='./tensor_board_logs', write_grads=True, write_graph=True,
                                           write_images=True)

history = model.fit(x_train, y_train, epochs=2500, validation_split=0.1,
                    callbacks=[reduce_lr, check_point, tensor_board], verbose=2,
                    batch_size=2000, shuffle=True, class_weight='auto')
# 保存最后一次训练的模型
model.save('model.h5')
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
