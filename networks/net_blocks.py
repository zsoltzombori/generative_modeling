from keras.layers import Dense, Reshape, Input, Lambda, Conv2D, Flatten, merge, Deconvolution2D, Activation, BatchNormalization
from keras.regularizers import l1, l2
from keras.layers.advanced_activations import LeakyReLU

leakyrelu_alpha = 0.2
batchnorm_momentum = 0.8

def dense_block(dims, wd, use_bn, activation):
    layers = []
    for dim in dims:
        layers.append(Dense(dim, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization(momentum=batchnorm_momentum)) 
        if activation == "leakyrelu":
            layers.append(LeakyReLU(alpha=leakyrelu_alpha))
        else:
            layers.append(Activation(activation))
    return layers

def conv_block(channels, kernel_size,  wd, use_bn, activation, strides=(1,1), padding='same'):
    layers = []
    for channel in channels:
        layers.append(Conv2D(channel, kernel_size, strides=strides, padding=padding, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization())
        if activation == "leakyrelu":
            layers.append(LeakyReLU(alpha=leakyrelu_alpha))
        else:
            layers.append(Activation(activation))
    return layers
