import numpy as np
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, GaussianNoise
from keras import backend as K
from keras.models import Model
from functools import partial

from util import *
import model_IO
import loss
import vis
import samplers
import util

import networks.models
from networks import dense, conv


def run(args, data):
    (x_train, x_test) = data

    # vanilla gan works better if images are scaled to [-1,1]
    # if you change this, make sure that the output of the generator is not a tanh
    x_train = (x_train * 2) - 1
    dim=int(np.sqrt(np.product(args.shape)))
    print(np.shape(x_train))
    if(args.model_type=="wgan-gp" or args.model_type=="wgan"):
        x_train=x_train.reshape(len(x_train),dim,dim,1)
    
    args['input_shape']=np.shape(x_train)[1:]
    
    print(np.shape(x_train)[1:])
    
    models, loss_features = build_models(args)
    assert set(("generator", "discriminator","gen_disc")) <= set(models.keys()), models.keys()

    # get losses
    loss_discriminator = loss.loss_factory(args.loss_discriminator, args, loss_features, combine_with_weights=True)
    loss_generator = loss.loss_factory(args.loss_generator, args, loss_features, combine_with_weights=True)
    metric_names = sorted(set(args.metrics + args.loss_discriminator + args.loss_generator))
    metrics = loss.loss_factory(metric_names, args, loss_features, combine_with_weights=False)
    

    # get optimizer
    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr)
    elif args.optimizer == "adam":
        if(args.model_type != "gan"):
            optimizer = Adam(lr=args.lr,beta_1=0., beta_2=0.9)
        else:
            optimizer = Adam(lr=args.lr, beta_1=0.5)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer
    
    # ===== Build DISCRIMINATOR =====
    models.generator.trainable = False
    # compile models
    if args.model_type=="wgan-gp":
        real_input = Input(args.original_shape)
        fake_input = Input(args.original_shape)
        interp_input = util.RandomWeightedAverage(args.batch_size)([real_input, fake_input])
        real_output = models.discriminator(real_input)
        fake_output = models.discriminator(fake_input)
        interp_output = models.discriminator(interp_input)
        discriminator = Model([real_input, fake_input], [real_output, fake_output, interp_output])
        partial_gp_loss = partial(loss.gp_loss, interpolated=interp_input)
        partial_gp_loss.__name__ = 'gp_loss'  # Functions need names or Keras will throw an error

        discriminator.compile(optimizer=optimizer,
            loss=[loss_discriminator,loss_discriminator,partial_gp_loss],
            metrics=metrics,
            loss_weights=[1, 1, 10])
    else:
        real_input = Input(args.original_shape)
        real_output=models.discriminator(real_input)
        
        discriminator=Model(real_input,real_output)
        discriminator.compile(optimizer=optimizer, loss=loss_discriminator, metrics=metrics)
    
    models.discriminator.trainable = False # For the combined model we will only train the generator
    models.generator.trainable = True
    
    # ===== Build GENERATOR =====
    z_gen = Input(shape=(args.latent_dim,))
    img = models.generator(z_gen)
    valid = models.discriminator(img)
    gen_disc = Model(z_gen, valid)
    gen_disc.compile(loss=loss_generator, optimizer=optimizer)

    print("===== Model type: "+args.model_type+" =====")
    print("Discriminator architecture:")
    print_model(discriminator)
    print("Generator architecture:")
    print_model(gen_disc)
    
    if(args.model_type=="gan"):
        valid_labels = np.ones((args.batch_size, 1))
        fake_labels = np.zeros((args.batch_size, 1))
    else:
        # Adversarial ground truths
        valid_labels = -np.ones((args.batch_size, 1))
        fake_labels = np.ones((args.batch_size, 1))
        dummy_y= np.zeros((args.batch_size, 1))

    sampler = samplers.sampler_factory(args)

    for step in range(args.training_steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        discriminator.trainable=True
        for l in discriminator.layers: l.trainable = True
        
        for i in range(args.gan_discriminator_update):
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], args.batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Generate a batch of new images
            gen_imgs = models.generator.predict(noise)

            # Train the discriminator
            if args.model_type=="wgan-gp":
                d_loss = discriminator.train_on_batch([imgs, gen_imgs], [valid_labels, fake_labels, dummy_y])
            else:
                d_loss1 = discriminator.train_on_batch(imgs,valid_labels)
                d_loss2 = discriminator.train_on_batch(gen_imgs, fake_labels)
                d_loss = 0.5 * (np.add(d_loss1, d_loss2))
            
            if(args.model_type=="wgan"):
                for l in discriminator.layers:
                   weights=l.get_weights()
                   weights=[np.clip(w,-0.01,0.01) for w in weights]
                   l.set_weights(weights)
                    
        discriminator.trainable=False
        for l in discriminator.layers: l.trainable = False
        # ---------------------
        #  Train Generator
        # ---------------------
        for i in range(args.gan_generator_update):
            noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = gen_disc.train_on_batch(noise, valid_labels)


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
    wgan_model=networks.models.iWGAN_01(args)
            
    if args.discriminator == "dense":
        discriminator = dense.build_model(args.original_shape, [1], args.discriminator_dims, args.discriminator_wd, args.discriminator_use_bn, args.activation, "sigmoid")
    elif args.discriminator == "conv":
        discriminator = conv.build_model(args.original_shape, [1], args.discriminator_conv_channels, args.discriminator_wd, args.discriminator_use_bn, args.activation, "sigmoid")
    elif (args.discriminator== "wgan_disc"):
        print("===============wgan disc=============="); 
        discriminator=wgan_model.build_discriminator(args.discriminator_use_bn);
    else:
        assert False, "Unrecognized value for discriminator: {}".format(args.discriminator)

    generator_input_shape = (args.latent_dim, )
    if args.generator == "dense":
        generator = dense.build_model(generator_input_shape, args.original_shape, args.generator_dims, args.generator_wd, args.generator_use_bn, args.activation, "tanh")
    elif (args.generator== "wgan_gen"):
        print("===============wgan gen=============="); 
        generator=wgan_model.build_generator(args.generator_use_bn);
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    gen_disc = Sequential([generator, discriminator])

    modelDict = AttrDict({})
    modelDict.gen_disc = gen_disc
    modelDict.discriminator = discriminator
    modelDict.generator = generator
    
    plot_model=False;
    if(plot_model):
        from keras.utils import plot_model
        plot_model(discriminator, 'model_disc.png',True,True,True)
        plot_model(generator, 'model_gen.png',True,True,True)
        
    return modelDict, loss_features


def generate_interpol_images(valid,fake):
    size=np.shape(valid)[0]
    alfa=np.random(size)
    
    return alfa*valid+(1-alfa)*fake
