import numpy as np
from keras.layers import Dense, Input, Activation, BatchNormalization, Flatten, Reshape
from keras.regularizers import l1, l2
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D,Flatten
from keras.layers import BatchNormalization, Dropout,ZeroPadding2D,Conv2DTranspose,Convolution2D,UpSampling2D
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU

import networks.net_blocks



class Improved_WGAN_paper_MNIST:
	def build_generator(latent_dim,batch_norm=False,imp_dim=784):
		DIM=64
		model = Sequential()
		
		# Dense:
		model.add(Dense(4*4*4*DIM,input_dim=latent_dim))
		
		# Reshape for Convolutional:
		if K.image_data_format() == 'channels_first':
			model.add(Reshape((4*DIM, 4, 4), input_shape=(4*4*4*DIM,)))
			bn_axis = 1
		else:
			model.add(Reshape((4, 4, 4*DIM), input_shape=(4*4*4*DIM,)))
			bn_axis = -1
		
		# First Convolutional:
		model.add(Conv2DTranspose(2*DIM, (5, 5), strides=2))
		if(batch_norm):
			model.add(BatchNormalization())
		model.add(Activation('relu'))
		
		## Reshape???
		
		# Second Convolutional:
		model.add(Conv2DTranspose(DIM, (5, 5), strides=2))
		if(batch_norm):
			model.add(BatchNormalization())
		model.add(Activation('relu'))
		
		# Third Convolutional:
		model.add(Conv2DTranspose(1, (5, 5), strides=2))
		if(batch_norm):
			model.add(BatchNormalization())
		model.add(Activation('sigmoid'))
		
		#model.add(Reshape((-1,28,28,1)))
		


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

	model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
	model.add(Reshape((7, 7, 128)))
	model.add(UpSampling2D())
	model.add(Conv2D(128, kernel_size=4, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))
	model.add(UpSampling2D())
	model.add(Conv2D(64, kernel_size=4, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation("relu"))
	model.add(Conv2D(1, kernel_size=4, padding="same"))
	model.add(Activation("tanh"))

	model.summary()

	noise = Input(shape=(latent_dim,))
	img = model(noise)

	return Model(noise, img)
	
def build_discriminator(input_shape):
	model = Sequential()
	
	model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
	model.add(ZeroPadding2D(padding=((0,1),(0,1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1))

	model.summary()

	img = Input(shape=input_shape)
	validity = model(img)

	return Model(img, validity)
