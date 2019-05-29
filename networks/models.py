import numpy as np
from keras.layers import Input, Activation, BatchNormalization, Flatten, Reshape
from keras.regularizers import l1, l2
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv2D, MaxPooling2D,Flatten
from keras.layers import BatchNormalization, Dropout,ZeroPadding2D,Conv2DTranspose,Convolution2D,UpSampling2D
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU
from keras import layers
import networks.net_blocks
#from keras.utils.vis_utils import plot_model


class Dense_model:
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
    
class iWGAN_01:
    def __init__(self,args):
        self.latent_dim=args.latent_dim
        self.args = args
        if('input_shape' in args):
            self.input_shape=args.input_shape
        else:
            self.input_shape=(28,28,1)
        
        if('generator_activation' not in args):
            self.gen_act="tanh"
        else:
            self.gen_act=args.generation_activation
        
        self.talky=True

        if(self.args.color):
            self.color = 3
        else:
            self.color = 1
        
        
        if("dropout" in args):
            self.dropout=(args.dropout=="True")
        else:
            self.dropout=False
        
        
    def build_generator(self,batch_norm=False):
        model = Sequential()
        s=self.input_shape[0]//4
        #shape = (self.input_shape[0], self.input_shape[1], self.color)
        color = self.color


        model.add(Dense(128 * s * s, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((s, s, 128)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        #model.add(UpSampling2D())
        model.add(Conv2D(color, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        model.add(Reshape(self.input_shape))
        
        if(self.talky):
            model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        #plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True)
        return Model(noise, img)
        
    def build_discriminator(self,batch_norm=False):
        model = Sequential()
        dim = np.prod(self.input_shape[:])
        s=self.input_shape[0]//4
        
        print('dim', dim)
        model.add(Reshape((-1, int(dim)), input_shape=self.input_shape))
        model.add(Dense(s*s*128, input_shape = self.input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((s, s, 128)))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        
        if(self.talky):
            model.summary()

        #plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True)

        img = Input(shape=self.input_shape)
        validity = model(img)

        return Model(img, validity)


class simple:
    def discriminator(self):
        net = Sequential()
        input_shape = (28, 28, 1)
        dropout_prob = 0.1
        #experiment.log_parameter('dis_dropout_prob', dropout_prob)

        net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        net.add(LeakyReLU())

        net.add(Conv2D(128, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))

        net.add(Conv2D(256, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))

        net.add(Conv2D(512, 5, strides=1, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))

        net.add(Flatten())
        net.add(Dense(1))
        #net.add(Activation('linear'))

        return net

    def generator(self):
        net = Sequential()
        dropout_prob = 0.1
        #experiment.log_parameter('adv_dropout_prob', dropout_prob)

        net.add(Dense(7*7*256, input_dim=100))
        net.add(BatchNormalization(momentum=0.9))
        net.add(LeakyReLU())
        net.add(Reshape((7,7,256)))
        net.add(Dropout(dropout_prob))

        net.add(UpSampling2D())
        net.add(Conv2D(128, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(LeakyReLU())

        net.add(UpSampling2D())
        net.add(Conv2D(64, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(LeakyReLU())

        net.add(Conv2D(32, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(LeakyReLU())

        net.add(Conv2D(1, 5, padding='same'))
        net.add(Activation('tanh'))

        return net
    
