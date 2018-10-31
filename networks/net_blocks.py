from keras.layers import Dense, Reshape, Input, Lambda, Convolution2D, Flatten, merge, Deconvolution2D, \
    Activation, BatchNormalization, Conv2DTranspose
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

# def conv_block(channels, kernelX, kernelY, wd, use_bn, activation, subsample, border_mode):
#     layers = []
#     for channel in channels:
#         layers.append(Convolution2D(channel, (kernelX, kernelY), strides=subsample, border_mode=border_mode, kernel_regularizer=l2(wd)))
#         if use_bn:
#             layers.append(BatchNormalization())
#         layers.append(Activation(activation))
#     return layers

def conv_block(filters, kernel, wd, use_bn, activation, strides, padding):
    layers = []
    for filter in filters:
        layers.append(Convolution2D(filter, kernel, strides, padding, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization())
        layers.append(Activation(activation))
    return layers

def deconv_block(filters, kernel, wd, use_bn, activation, strides, padding):
    layers = []
    for filter in filters:
        layers.append(Conv2DTranspose(filter, kernel, strides, padding, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization())
        layers.append(Activation(activation))
    return layers



