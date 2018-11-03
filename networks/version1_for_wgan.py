import numpy as np
from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten, Reshape
from keras.regularizers import l1, l2
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D,Flatten
from keras.layers import BatchNormalization, Dropout,ZeroPadding2D,Conv2DTranspose,Convolution2D
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU

import networks.net_blocks



class Improved_WGAN_paper_MNIST:
	def build_generator(latent_dim,batch_norm=False,imp_dim=784):
		DIM=inp_dim
		model = Sequential()
		model.add(Dense(4*4*4*DIM,input_dim=latent_dim))
		if(batch_norm):
			model.add(BatchNormalization())
		modell.add(ReLu())
		#model.add(Reshape(!*DIM
















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
    
def build_generator(latent_dim,linear=False,batch_norm=True):
	model = Sequential()
	model.add(Dense(1024, input_dim=100))
	model.add(LeakyReLU())
	model.add(Dense(128 * 7 * 7))
	if(batch_norm):
		model.add(BatchNormalization())
	model.add(LeakyReLU())
	if K.image_data_format() == 'channels_first':
		model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
		bn_axis = 1
	else:
		model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
		bn_axis = -1
	model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
	if(batch_norm):
		model.add(BatchNormalization(axis=bn_axis))
	model.add(LeakyReLU())
	model.add(Convolution2D(64, (5, 5), padding='same'))
	if(batch_norm):
		model.add(BatchNormalization(axis=bn_axis))
	model.add(LeakyReLU())
	model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
	if(batch_norm):
		model.add(BatchNormalization(axis=bn_axis))
	model.add(LeakyReLU())
	# Because we normalized training inputs to lie in the range [-1, 1],
	# the tanh function should be used for the output of the generator to ensure its output
	# also lies in this range.
	if(linear):
		model.add(Convolution2D(1, (5, 5), padding='same', activation='linear'))
	else:
		model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
	inp=Input(shape=(latent_dim,))
	out=model(inp)
	m=Model(inp,out)
	return m
	
def build_discriminator(input_shape):
	model = Sequential()
	if K.image_data_format() == 'channels_first':
		model.add(Conv2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
	else:
		model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
	model.add(LeakyReLU())
	model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
	model.add(LeakyReLU())
	model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
	model.add(LeakyReLU())
	model.add(Flatten())
	model.add(Dense(1024, kernel_initializer='he_normal'))
	model.add(LeakyReLU())
	model.add(Dense(1, kernel_initializer='he_normal'))

	img = Input(shape=input_shape)
	validity = model(img)
	m=Model(img, validity)
	return m
