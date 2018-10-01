import numpy as np
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, GaussianNoise
from keras import backend as K
from keras.models import Model

from util import *
import model_IO
import loss
import vis
import samplers

import networks.dense


def run(args, data):
    (x_train, x_test) = data

    models, loss_features = build_models(args)
    assert set(("generator", "discriminator", "gen_disc")) <= set(models.keys()), models.keys()

    print("Discriminator architecture:")
    print_model(models.discriminator)
    print("Generator architecture:")
    print_model(models.generator)

    # get losses
    loss_discriminator = loss.loss_factory(args.loss_discriminator, args, loss_features, combine_with_weights=True)
    loss_generator = loss.loss_factory(args.loss_generator, args, loss_features, combine_with_weights=True)
    metric_names = sorted(set(args.metrics + args.loss_discriminator + args.loss_generator))
    metrics = loss.loss_factory(metric_names, args, loss_features, combine_with_weights=False)
    

    # get optimizer
    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    # compile models
    print(metrics)
    print(metric_names)
    models.discriminator.compile(optimizer=optimizer, loss=loss_discriminator, metrics=metrics)
    models.discriminator.trainable = False # For the combined model we will only train the generator
    models.gen_disc.compile(optimizer=optimizer, loss=loss_generator)

    disc_losses = []
    gen_losses = []

    # Adversarial ground truths
    valid = np.ones((args.batch_size, 1))
    fake = np.zeros((args.batch_size, 1))

    for step in range(args.training_steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of images
        idx = np.random.randint(0, x_train.shape[0], args.batch_size)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        # Generate a batch of new images
        gen_imgs = models.generator.predict(noise)

        # Train the discriminator
        d_loss_real = models.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = models.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        # Train the generator (to have the discriminator label samples as valid)
        g_loss = models.gen_disc.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step, d_loss[0], 100*d_loss[1], g_loss))

        disc_losses.append(d_loss[0])
        gen_losses.append(g_loss)


    # # save models
    # model_IO.save_autoencoder(models, args)

    # # display randomly generated images
    # sampler = samplers.sampler_factory(args)
    # vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))

    # # display one batch of reconstructed images
    # vis.displayReconstructed(x_train[:args.batch_size], args, models, "{}/train".format(args.outdir))
    # vis.displayReconstructed(x_test[:args.batch_size], args, models, "{}/test".format(args.outdir))


    # # display image interpolation
    # vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix,
    #                   anchor_indices = data_object.anchor_indices, toroidal=args.toroidal)

    # vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
    # vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))



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
                                               "linear")
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    gen_disc = Sequential([generator, discriminator])

    modelDict = AttrDict({})
    modelDict.gen_disc = gen_disc
    modelDict.discriminator = discriminator
    modelDict.generator = generator

    return modelDict, loss_features
