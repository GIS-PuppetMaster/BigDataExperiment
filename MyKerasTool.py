def Dense_BN(input, units=8, bn_axis=-1, activation='tanh'):
    from keras.layers import Activation, BatchNormalization, Dense, Concatenate, Dropout
    from keras import regularizers
    dense = Dense(units)(input)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    return Activation(activation)(dense)


def Dense_res_block3(input, layercell=(32, 16)):
    from keras.layers import Activation, BatchNormalization, Dense, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('tanh')(bn)
    dense01 = Dense(layercell[0], kernel_regularizer=regularizers.l2(0.01))(ac)
    dense01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
    dense01_ac = Activation('tanh')(dense01_bn)
    dense02 = Dense(layercell[1], kernel_regularizer=regularizers.l2(0.01))(dense01_ac)
    dense02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
    dense02_ac = Activation('tanh')(dense02_bn)
    dense03 = Dense(input.shape.as_list()[1], kernel_regularizer=regularizers.l2(0.01))(dense02_ac)
    merge = Add()([dense03, input])
    return merge


def CuDNNLSTM_identity_block2(input, size=(32,), block_name='CuDNNLSTM_identity_block',pre_activation=True):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    """
    恒等
    """
    if pre_activation:
        bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'bn')(input)
        ac = Activation('tanh', name=block_name+'ac')(bn)
        lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'lstm01')(ac)
    else:
        lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'lstm01')(input)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'lstm01_bn')(lstm01)
    ac = Activation('tanh', name=block_name+'lstm01_ac')(bn)
    lstm02 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'lstm02')(ac)
    merge = Add()([lstm02, input])
    if not pre_activation:
        merge = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name + 'bn')(merge)
        merge = Activation('tanh', name=block_name + 'ac')(merge)
    return merge

def CuDNNLSTM_lstm_block2(input, size=(32,16), block_name='CuDNNLSTM_lstm_block'):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    """
    降维
    """
    lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'lstm01')(input)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'lstm01_bn')(lstm01)
    ac = Activation('tanh', name=block_name+'lstm01_ac')(bn)
    short_cut = CuDNNLSTM(size[1], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'shortcut')(input)
    short_cut = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'shortcut_bn')(short_cut)
    short_cut = Activation('tanh', name=block_name+'shortcut_ac')(short_cut)
    lstm02 = CuDNNLSTM(size[1], kernel_regularizer=regularizers.l2(0.01), return_sequences=True, name=block_name+'lstm02')(ac)
    lstm02 = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'lstm02_bn')(lstm02)
    lstm02 = Activation('tanh', name=block_name+'lstm02_ac')(lstm02)
    merge = Add()([lstm02, short_cut])
    return merge

"""
def CuDNNLSTM_res_block3(input, size=(32, 32)):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, CuDNNLSTM, Bidirectional, Add
    from keras import regularizers
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input)
    ac = Activation('tanh')(bn)
    lstm01 = CuDNNLSTM(size[0], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm01)
    ac = Activation('tanh')(bn)
    lstm02 = CuDNNLSTM(size[1], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    bn = BatchNormalization(epsilon=1e-4, scale=True, center=True)(lstm02)
    ac = Activation('tanh')(bn)
    lstm03 = CuDNNLSTM(input.shape.as_list()[2], kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(ac)
    merge = Add()([lstm03, input])
    return merge
"""

def Conv1D_identity_block(input, filters=(3, 3, 3), kernel_size=3,
                      padding=('valid', 'same', 'valid'),
                      data_format='channels_first',
                      activation=('tanh', 'tanh', 'tanh'), block_name='Conv1D_identity_block'):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D, Permute, Dropout
    import keras
    shortcut = input
    conv01 = Conv1D(filters=filters[0], kernel_size=1, padding=padding[0], strides=1, data_format=data_format, name=block_name+'conv01')(input)
    conv01 = Dropout(0.2)(conv01)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv01_bn')(conv01)
    conv01_ac = Activation(activation[0], name=block_name+'conv01_ac')(conv01_bn)

    conv02 = Conv1D(filters=filters[1], kernel_size=kernel_size, padding=padding[1], strides=1, data_format=data_format, name=block_name+'conv02')(conv01_ac)
    conv02 = Dropout(0.2)(conv02)
    conv02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv02_bn')(conv02)
    conv02_ac = Activation(activation[1], name=block_name+'conv02_ac')(conv02_bn)

    conv03 = Conv1D(filters=filters[2], kernel_size=1, padding=padding[2], strides=1, data_format=data_format, name=block_name+'conv03')(conv02_ac)
    conv03 = Dropout(0.2)(conv03)
    conv03_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv03_bn')(conv03)
    merge03 = Add(name=block_name+'merge03_Add')([conv03_bn, shortcut])
    conv03_ac = Activation(activation[2], name=block_name+'conv03_ac')(merge03)
    return conv03_ac


def Conv1D_conv_block(input, filters=(3, 3, 3), kernel_size=3, strides=1,
                      padding=('valid', 'same', 'valid', 'valid'),
                      data_format='channels_first',
                      activation=('tanh', 'tanh', 'tanh'), block_name='Conv1D_conv_block'):
    from keras.layers import Conv1D, Activation, BatchNormalization, Concatenate, Add, ZeroPadding1D, Permute, Dropout
    import keras
    shortcut = input
    conv01 = Conv1D(filters=filters[0], kernel_size=1, padding=padding[0], strides=strides, data_format=data_format, name=block_name+'conv01')(input)
    conv01 = Dropout(0.2)(conv01)
    conv01_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv01_bn')(conv01)
    conv01_ac = Activation(activation[0], name=block_name+'conv01_ac')(conv01_bn)

    conv02 = Conv1D(filters=filters[1], kernel_size=kernel_size, padding=padding[1], strides=1,
                    data_format=data_format, name=block_name+'conv02',kernel_regularizer=keras.regularizers.l2(0.001))(conv01_ac)
    conv02 = Dropout(0.2)(conv02)
    conv02_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv02_bn')(conv02)
    conv02_ac = Activation(activation[1], name=block_name+'conv02_ac')(conv02_bn)

    conv03 = Conv1D(filters=filters[2], kernel_size=1, padding=padding[2], strides=1, data_format=data_format, name=block_name+'conv03')(conv02_ac)
    conv03 = Dropout(0.2)(conv03)
    conv03_bn = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'conv03_bn')(conv03)

    shortcut = Conv1D(filters=filters[2], kernel_size=1, padding=padding[3], strides=strides, data_format=data_format, name=block_name+'shortcut_conv')(shortcut)
    shortcut = BatchNormalization(epsilon=1e-4, scale=True, center=True, name=block_name+'shortcut_bn')(shortcut)
    merge03 = Add(name=block_name+'merge03_Add')([conv03_bn, shortcut])
    conv03_ac = Activation(activation[2], name=block_name+'conv03_ac')(merge03)
    return conv03_ac

def Dense_layer_connect(input, size, units=8):
    from keras.layers import Concatenate,Dense,BatchNormalization,Activation,Flatten,Reshape
    from keras import regularizers
    # 升维，增加深度轴
    input_ = Reshape((1, size))(input)
    dense0 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(input_)
    dense0_bn = BatchNormalization(axis=1, epsilon=1e-4, scale=True, center=True)(dense0)
    dense0 = Activation('tanh')(dense0_bn)

    dense1 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense0)
    dense1_bn = BatchNormalization(axis=1, epsilon=1e-4, scale=True, center=True)(dense1)
    dense1 = Activation('tanh')(dense1_bn)

    dense2 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2_bn = BatchNormalization(axis=1, epsilon=1e-4, scale=True, center=True)(dense2)
    # 在深度轴上合并
    add2 = Concatenate(axis=1)([dense0_bn, dense2_bn])
    dense2 = Activation('tanh')(add2)

    dense3 = Dense(units, kernel_regularizer=regularizers.l2(0.01))(dense2)
    dense3_bn = BatchNormalization(axis=1, epsilon=1e-4, scale=True, center=True)(dense3)
    add3 = Concatenate(axis=1)([dense0_bn, dense1_bn, dense3_bn])
    dense3 = Activation('tanh')(add3)
    # 降维展平
    flatten = Flatten()(dense3)
    return flatten