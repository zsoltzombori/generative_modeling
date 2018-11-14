import numpy as np
from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten, Reshape, Convolution2D
from keras.regularizers import l1, l2
from keras.models import Model

import networks.net_blocks

def build_model(input_shape, output_shape, dims, wd, use_bn, activation, last_activation):
    inputs = Input(shape=input_shape)
    outputs = inputs

    if len(inputs.shape) > 2:
        outputs = Flatten()(inputs)

    layers = networks.net_blocks.dense_block(dims, wd, use_bn, activation)
    for l in layers:
        outputs = l(outputs)

    outputs = Dense(np.prod(output_shape), activation=last_activation)(outputs)
    outputs = Reshape(output_shape)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_conv_model(input_shape, intermediate_dim, output_shape,
                     filters, kernel, wd, use_bn, activation, strides, padding, last_activation):
    inputs = Input(shape = input_shape)
    outputs = inputs

    layers = networks.net_blocks.conv_block(filters, kernel, wd, use_bn,
                                            activation, strides, padding)
    for l in layers:
        outputs = l(outputs)

    outputs = Flatten()(outputs)

    outputs = Dense(intermediate_dim, activation='relu')(outputs)

    outputs = Dense(np.prod(output_shape), activation=last_activation)(outputs)
    outputs = Reshape(output_shape)(outputs)

    convmodel = Model(inputs = inputs, outputs = outputs)

    return convmodel


def build_deconv_model(input_shape, output_wh, filters, kernel, wd, use_bn, activation, strides, padding):

    # input_dim = build_conv_model().output_shape

    # inputs = build_conv_model()
    inputs = Input(shape = input_shape)
    outputs = inputs

#    outputs = Dense(input_dim, activation = 'relu')(outputs)
    outputs = Dense(filters[0] * np.prod(output_wh), activation='relu')(outputs)
    outputs = Reshape(list(output_wh) + [filters[0]])(outputs)

    layers = networks.net_blocks.deconv_block(filters[1:], kernel, wd, use_bn,
                                            activation, strides, padding)
    for l in layers:
        outputs = l(outputs)

    outputs = Convolution2D(1, (2,2), padding='same', activation='sigmoid')(outputs)

    deconvmodel = Model(inputs = inputs, outputs = outputs)

    return deconvmodel

