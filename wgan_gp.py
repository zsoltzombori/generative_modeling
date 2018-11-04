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


import networks.dense
import networks.version1_for_wgan

BATCH_SIZE=64

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
        

def run(args, data):
    (x_train, x_test) = data
    
    BATCH_SIZE=args.batch_size

    # vanilla gan works better if images are scaled to [-1,1]
    # if you change this, make sure that the output of the generator is not a tanh
    x_train = (x_train * 2) - 1
    
    # build the models
    models, loss_features = build_models(args)
    assert set(("generator", "discriminator", "gen_disc")) <= set(models.keys()), models.keys()

    print("Discriminator architecture:")
    print_model(models.discriminator)
    print("Generator architecture:")
    print_model(models.generator)

    ##### Build GENERATOR: #####
    for layer in models.discriminator.layers:
        layer.trainable = False
    models.discriminator.trainable = False
    
    generator_input = Input(shape=(100,))
    generator_layers = models.generator(generator_input)
    discriminator_layers_for_generator = models.discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    
    ##### BUILD DISCRIMINATOR #####
    
    for layer in models.discriminator.layers:
        layer.trainable = True
    for layer in models.generator.layers:
        layer.trainable = False
    models.discriminator.trainable = True
    models.generator.trainable = False
    
    real_samples = Input(shape=x_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = models.generator(generator_input_for_discriminator)
    discriminator_output_from_generator = models.discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = models.discriminator(real_samples)
    
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = models.discriminator(averaged_samples)
    
    GRADIENT_PENALTY_WEIGHT=10
    partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    
    
    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
                          
    print("Discriminator architecture:")
    print_model(discriminator_mode)
    print("Generator architecture:")
    print_model(generator_model)
    
    positive_y = np.ones((args.batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((args.batch_size, 1), dtype=np.float32)
    
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

            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)

            d_loss = discriminator_model.train_on_batch([imgs, noise],[positive_y, negative_y, dummy_y])
                  
        #models.discriminator.trainable=False;
        # ---------------------
        #  Train Generator
        # ---------------------
        for i in range(args.gan_generator_update):
            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = generator_model.train_on_batch(noise, positive_y)


        # Plot the progress
        if (step+1) % args.frequency == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step+1, d_loss[0], 100*d_loss[1], g_loss))
            vis.displayRandom((10, 10), args, models, sampler, "{}/random-{}".format(args.outdir, step+1))


    # save models
    model_IO.save_gan(models, args)

    # display randomly generated images
    vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))


def build_models(args):
    loss_features = AttrDict({})
            
    if args.discriminator == "dense":
        discriminator = networks.dense.build_model(args.original_shape,
                                                   [1],
                                                   args.discriminator_dims,
                                                   args.discriminator_wd,
                                                   args.discriminator_use_bn,
                                                   args.activation,
                                                   "sigmoid")
    elif (args.discriminator== "wgan_disc"):
        print("===============wgan disc=============="); 
        discriminator=networks.version1_for_wgan.build_discriminator((28,28,1));
    else:
        assert False, "Unrecognized value for discriminator: {}".format(args.discriminator)

    generator_input_shape = (args.latent_dim, )
    if args.generator == "dense":
        generator = networks.dense.build_model(generator_input_shape,
                                               args.original_shape,
                                               args.generator_dims,
                                               args.generator_wd,
                                               args.generator_use_bn,
                                               args.activation,
                                               "tanh")
    elif (args.generator== "wgan_gen"):
        print("===============wgan gen=============="); 
        generator=networks.version1_for_wgan.build_generator(args.latent_dim,args.linear,False);
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    gen_disc = Sequential([generator, discriminator])

    modelDict = AttrDict({})
    modelDict.gen_disc = gen_disc
    modelDict.discriminator = discriminator
    modelDict.generator = generator

    return modelDict, loss_features


def generate_interpol_images(valid,fake):
    size=np.shape(valid)[0]
    alfa=np.random(size)
    
    return alfa*valid+(1-alfa)*fake
