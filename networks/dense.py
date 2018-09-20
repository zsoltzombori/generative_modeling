from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten
from keras.regularizers import l1, l2
from keras.models import Model

import networks.net_blocks

def build_model(input_shape, dims, wd, use_bn, activation):
    inputs = Input(shape=input_shape)
    outputs = inputs

    if len(inputs.shape) > 2:
        outputs = Flatten()(inputs)

    layers = networks.net_blocks.dense_block(dims, wd, use_bn, activation)
    print("xxx, ", len(layers))
    for l in layers:
        outputs = l(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model
    
