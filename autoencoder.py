import numpy as np
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, Concatenate, Flatten
from keras import backend as K
from keras.models import Model

from util import AttrDict, print_model_shapes
import model_IO
import loss
import vis
import samplers

import networks.dense


def run(args, data):
    ((x_train, y_train), (x_test, y_test)) = data

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "generator")) <= set(models.keys()), models.keys()
    
    print("Encoder architecture:")
    print_model_shapes(models.encoder)
    print("Generator architecture:")
    print_model_shapes(models.generator)

    # get losses
    losses, metrics = loss.loss_factory(args, loss_features)

    # get optimizer
    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    # compile autoencoder
    models.ae.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    # TODO specify callbacks
    cbs = []

    # train the autoencoder
    models.ae.fit((x_train, y_train), x_train,
                  verbose=args.verbose,
                  shuffle=True,
                  epochs=args.nb_epoch,
                  batch_size=args.batch_size,
                  callbacks = cbs)
                  # validation_data=((x_test, y_test), x_test)
    # )

    # save models
    model_IO.save_autoencoder(models, args)

    # display randomly generated images
    sampler = samplers.sampler_factory(args)
    vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))

    # display one batch of reconstructed images
    vis.displayReconstructed(x_train[:args.batch_size], args, models, "{}/train".format(args.outdir))
    vis.displayReconstructed(x_test[:args.batch_size], args, models, "{}/test".format(args.outdir))


    # # display image interpolation
    # vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix,
    #                   anchor_indices = data_object.anchor_indices, toroidal=args.toroidal)

    # vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
    # vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))



def build_models(args):
    loss_features = AttrDict({})

    input_x = Input(shape=args.original_shape)
    input_y = Input(shape=(args.y_label_count,))
    concat_layer = Concatenate()
    merged = concat_layer((Flatten()(input_x), input_y))
    merge_model = Model(inputs=[input_x, input_y], output = merge)
    concatenated_input_size = np.prod(args.original_shape) + args.y_label_count

    input_y2 = Input(shape=args.y_label_count)
    input_latent = Input(shape=args.latent_dim)
    concat_layer2 = Concatenate()
    merged_latent = concat_layer((input_latent, input_y2))
    merge_latent_model = Model(inputs=[input_latent, input_y2], output = merged_latent)

    if args.sampling:
        encoder_output_shape = (args.latent_dim, 2)
    else:
        encoder_output_shape = (args.latent_dim, )
        
    if args.encoder == "dense":
        encoder = networks.dense.build_model(args.original_shape,
                                             encoder_output_shape,
                                             args.encoder_dims,
                                             args.encoder_wd,
                                             args.encoder_use_bn,
                                             args.activation,
                                             "linear")
        encoder = Sequential([merge_model, encoder])
    else:
        assert False, "Unrecognized value for encoder: {}".format(args.encoder)

    generator_input_shape = (args.latent_dim + args.y_label_count, )
    if args.generator == "dense":
        generator = networks.dense.build_model(generator_input_shape,
                                               args.original_shape,
                                               args.generator_dims,
                                               args.generator_wd,
                                               args.generator_use_bn,
                                               args.activation,
                                               "linear")
        generator = Sequential([merge_latent_model, generator])
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    if args.sampling:
        sampler_model = add_gaussian_sampling(encoder_output_shape, args)

        inputs = Input(shape=args.original_shape)
        hidden = encoder(inputs)
        (z, z_mean, z_log_var) = sampler_model(hidden)
        encoder = Model(inputs, [z, z_mean, z_log_var])
        
        loss_features["z_mean"] = z_mean
        loss_features["z_log_var"] = z_log_var
        output = generator(z, input_y)
        ae = Model(inputs=[input_x, input_y], outputs=output)
    else:
        ae = Sequential([encoder, generator])

    modelDict = AttrDict({})
    modelDict.ae = ae
    modelDict.encoder = encoder
    modelDict.generator = generator

    return modelDict, loss_features


def add_gaussian_sampling(input_shape, args):
    assert input_shape[-1] == 2
    inputs = Input(shape=input_shape)

    z_mean = Lambda(lambda x: x[...,0], output_shape=input_shape[:-1])(inputs)
    z_log_var = Lambda(lambda x: x[...,1], output_shape=input_shape[:-1])(inputs)
    
    output_shape = list(K.int_shape(z_mean))
    output_shape[0] = args.batch_size
    
    def sampling(inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=output_shape, mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = Lambda(sampling)([z_mean, z_log_var])
    sampler_model = Model(inputs, [z, z_mean, z_log_var])
    return sampler_model
