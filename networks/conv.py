import numpy as np
from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.regularizers import l1, l2
from keras.models import Model

import networks.net_blocks

def build_model(input_shape, output_shape, channels, wd, use_bn, activation, last_activation):
    inputs = Input(shape=input_shape)
    outputs = inputs

    assert len(input_shape) == 3 or len(output_shape) == 3, "Either the input or the output shape of this conv builder should be 3 dim (image)"

    if len(input_shape) != 3: # we are mapping a vector to an image
        assert len(input_shape) == 1 
        out_x, out_y, out_ch = output_shape
        outputs = Dense(out_x * out_y)(outputs)
        outputs = Reshape((out_x, out_y, 1))(outputs)
        
    layers = networks.net_blocks.conv_block(channels, 3, wd, use_bn, activation)
    for l in layers:
        outputs = l(outputs)

    if len(input_shape) != 3: # we are mapping a vector to an image
        outputs = Conv2D(out_ch, 3, strides=(1,1), padding='same', kernel_regularizer=l2(wd))(outputs)
    else: # we are mapping an image to a vector
        outputs = Flatten()(outputs)
        outputs = Dense(np.prod(output_shape), activation=last_activation)(outputs)
        outputs = Reshape(output_shape)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_model_conv_encoder(input_shape, output_shape, channels, wd, use_bn, activation, last_activation):
    inputs = Input(shape=input_shape)
    outputs = inputs

    outputs = Conv2D(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2D(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2D(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2D(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(256)(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Dense(256)(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Dense(np.prod(output_shape), activation=last_activation)(outputs)
    outputs = Reshape(output_shape)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_model_conv_decoder(input_shape, output_shape, channels, wd, use_bn, activation, last_activation):
    inputs = Input(shape=input_shape)
    outputs = inputs

    outputs = Dense(256)(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Dense(np.prod(4*4*32))(outputs)
    outputs = Activation(activation)(outputs)

    outputs = Reshape((4, 4, 32))(outputs)

    outputs = Conv2DTranspose(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2DTranspose(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2DTranspose(32, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Activation(activation)(outputs)
    outputs = Conv2DTranspose(1, 4, strides=(2,2), padding='same', kernel_regularizer=l2(wd))(outputs)
    outputs = Reshape(output_shape)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model

