import numpy as np
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, GaussianNoise
from keras import backend as K
from keras.models import Model
from functools import partial
from keras.layers.merge import _Merge

from util import *
import model_IO
import loss
import vis
import samplers
import util

from keras.datasets import mnist


import networks.dense
import networks.models

class RandomWeightedAverage(_Merge):
    def __init__(self,b_size):
        _Merge.__init__(self)
        self.BATCH_SIZE=b_size
    def _merge_function(self, inputs):
        weights = K.random_uniform((self.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
        
def get_cars():
    import random
    import glob, os
    import matplotlib.image as mpimg

    x_train=[]

    dir="./datasets/cars_28/"
    os.chdir(dir)
    for filename in glob.glob("*.jpg"):
        img=mpimg.imread(filename)
        img=np.resize(img,(28,28,1))
        
        x_train.append(img)
    os.chdir("../")

    x_train = np.array(x_train).astype('float32') / 255.
    print("Train:",np.shape(x_train))
    return x_train
    
def run(args, data):
    #(x_train, x_test) = data
    
    #(x_train, _), (_, _) = mnist.load_data()
    #x_train=x_train.reshape((-1,784,1))
    x_train=get_cars()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    #x_train = np.expand_dims(x_train, axis=3)
    args['input_shape']=np.shape(x_train)[1:]
    print(np.shape(x_train))
    
    # build the models
    models, loss_features = build_models(args)
    assert set(("generator", "critic")) <= set(models.keys()), models.keys()

    print("Discriminator architecture:")
    print_model(models.critic)
    print("Generator architecture:")
    print_model(models.generator)
    
    optimizer = RMSprop(lr=0.00005)
    
    ## Disc: ##
    models.generator.trainable = False
    real_img = Input(shape=np.shape(x_train)[1:])
    z_disc = Input(shape=(args.latent_dim,))
    fake_img = models.generator(z_disc)

    fake = models.critic(fake_img)
    valid = models.critic(real_img)
    
    interpolated_img = RandomWeightedAverage(args.batch_size)([real_img, fake_img])
    validity_interpolated = models.critic(interpolated_img)
    
    partial_gp_loss = partial(gradient_penalty_loss,
                      averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    critic_model = Model(inputs=[real_img, z_disc],
                        outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss],
                                    optimizer=optimizer,
                                    loss_weights=[1, 1, 10])
                                    
    ## Generator ##
    models.critic.trainable = False
    models.generator.trainable = True
    
    z_gen = Input(shape=(100,))
    img = models.generator(z_gen)
    valid = models.critic(img)
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

    valid = -np.ones((args.batch_size, 1))
    fake =  np.ones((args.batch_size, 1))
    dummy = np.zeros((args.batch_size, 1))
    
    sampler = samplers.sampler_factory(args)
    
    for step in range(args.training_steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        #models.discriminator.trainable=True
        for i in range(args.gan_discriminator_update):
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], args.batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            
            d_loss = critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])
            
        #models.discriminator.trainable=False;
        # ---------------------
        #  Train Generator
        # ---------------------
        for i in range(args.gan_generator_update):
            #noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = generator_model.train_on_batch(noise, valid)


        # Plot the progress
        if (step+1) % args.frequency == 0:
            print ("%d [D loss: %f , acc.: %.2f%%] [G loss: %f]" % (step+1, d_loss[0], 100*d_loss[1], g_loss))
            vis.displayRandom((10, 10), args, models, sampler, "{}/random-{}".format(args.outdir, step+1))


    # save models
    model_IO.save_gan(models, args)

    # display randomly generated images
    vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))


def build_models(args):
    loss_features = AttrDict({})
    
    ### Fill missing args: ###
    
    wgan_model=networks.models.iWGAN_01(args)
    critic=wgan_model.build_discriminator(True)

    generator=wgan_model.build_generator(True)

    modelDict = AttrDict({})
    modelDict.critic= critic
    modelDict.generator = generator

    return modelDict, loss_features


def generate_interpol_images(valid,fake):
    size=np.shape(valid)[0]
    alfa=np.random(size)
    
    return alfa*valid+(1-alfa)*fake
