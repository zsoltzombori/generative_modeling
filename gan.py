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
import networks.version1_for_wgan


def run(args, data):
    (x_train, x_test) = data

    # vanilla gan works better if images are scaled to [-1,1]
    # if you change this, make sure that the output of the generator is not a tanh
    x_train = (x_train * 2) - 1
    
    print(np.shape(x_train))
    x_train=x_train.reshape(-1,28,28,1)
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
        optimizer = Adam(lr=args.lr, epsilon=0.5)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    # compile models
    models.discriminator.compile(optimizer=optimizer, loss=loss_discriminator, metrics=metrics)
    models.discriminator.trainable = False # For the combined model we will only train the generator
    models.gen_disc.compile(optimizer=optimizer, loss=loss_generator)


    print(args.model_type)
    # Adversarial ground truths
    if(args.model_type=="wgan"):
        valid_labels = np.ones((args.batch_size, 1))
        fake_labels = -np.ones((args.batch_size, 1))
    else:
        valid_labels = np.ones((args.batch_size, 1))
        fake_labels = np.zeros((args.batch_size, 1))

    assert args.batch_size % 2 == 0
    half = args.batch_size // 2
    if(args.model_type!="wgan"):
        out_batch1 = np.concatenate([valid_labels[:half], fake_labels[:half]])
        out_batch2 = np.concatenate([valid_labels[half:], fake_labels[half:]])
    else:
        out_batch1=valid_labels
        out_batch2=fake_labels
        
    sampler = samplers.sampler_factory(args)

    for step in range(args.training_steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        models.discriminator.trainable=True
        for l in models.discriminator.layers: l.trainable = True
        
        for i in range(args.gan_discriminator_update):
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], args.batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Generate a batch of new images
            gen_imgs = models.generator.predict(noise)

            if(args.model_type!="wgan"):
                # mix the two batches
                in_batch1 = np.concatenate([imgs[:half], gen_imgs[:half]])
                in_batch2 = np.concatenate([imgs[half:], gen_imgs[half:]])
            else:
                in_batch1=imgs
                in_batch2=gen_imgs
            # Train the discriminator
            d_loss1 = models.discriminator.train_on_batch(in_batch1, out_batch1)
            d_loss2 = models.discriminator.train_on_batch(in_batch2, out_batch2)
            d_loss = 0.5 * (np.add(d_loss1, d_loss2))
            
            if(args.model_type=="wgan"):
                for l in models.discriminator.layers:
                    weights=l.get_weights()
                    weights=[np.clip(w,-0.01,0.01) for w in weights]
                    l.set_weights(weights)
                    
        models.discriminator.trainable=False
        # ---------------------
        #  Train Generator
        # ---------------------
        for i in range(args.gan_generator_update):
            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = models.gen_disc.train_on_batch(noise, valid_labels)


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
        generator=networks.version1_for_wgan.build_generator(args.latent_dim);
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    gen_disc = Sequential([generator, discriminator])

    modelDict = AttrDict({})
    modelDict.gen_disc = gen_disc
    modelDict.discriminator = discriminator
    modelDict.generator = generator

    return modelDict, loss_features
