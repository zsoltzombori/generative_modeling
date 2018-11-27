import numpy as np
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, Concatenate, Flatten
from keras import backend as K
from keras.models import Model

from util import *
import model_IO
import loss
import vis
import samplers
import callbacks

from networks import dense, conv


def run(args, data):
    ((x_train, y_train), (x_test, y_test)) = data

    sampler = samplers.sampler_factory(args)

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "generator")) <= set(models.keys()), models.keys()

    print("Encoder architecture:")
    print_model(models.encoder)
    print("Generator architecture:")
    print_model(models.generator)

    # get losses
    loss_names = sorted(set(args.loss_encoder + args.loss_generator))
    losses = loss.loss_factory(loss_names, args, loss_features, combine_with_weights=True)
    metric_names = sorted(set(args.metrics + tuple(loss_names)))
    metrics = loss.loss_factory(metric_names, args, loss_features, combine_with_weights=False)

    sampler = samplers.sampler_factory(args)

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
    cbs.append(callbacks.ImageDisplayCallback((x_train, y_train), (x_test, y_test), args, models, sampler))

    # train the autoencoder
    models.ae.fit([x_train, y_train], x_train,
                  verbose=args.verbose,
                  shuffle=True,
                  epochs=args.nb_epoch,
                  batch_size=args.batch_size,
                  callbacks = cbs,
                  validation_data=[[x_test, y_test], x_test])
    # )

    # save models
    model_IO.save_autoencoder(models, args)

    # display randomly generated images
    # vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))

    # display one batch of reconstructed images
    # vis.displayReconstructed(x_train[:args.batch_size], args, models, "{}/train".format(args.outdir))
    # vis.displayReconstructed(x_test[:args.batch_size], args, models, "{}/test".format(args.outdir))


    # # display image interpolation
    # vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix,
    #                   anchor_indices = data_object.anchor_indices, toroidal=args.toroidal)

    # vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
    # vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))


def build_models(args):
    loss_features = AttrDict({})

    input_x = Input(shape=args.original_shape)
    input_y = Input(shape=(args.y_label_count, ))
    merged = Concatenate()([Flatten()(input_x), input_y])
    merge_model = Model(inputs=[input_x, input_y], outputs = merged)
    concatenated_input_size = np.prod(args.original_shape) + args.y_label_count

    input_latent = Input(shape=(args.latent_dim, ))
    input_y2 = Input(shape=(args.y_label_count, ))
    merged_latent = Concatenate()([input_latent, input_y2])
    merge_latent_model = Model(inputs=[input_latent, input_y2], outputs = merged_latent)

    if args.sampling:
        encoder_output_shape = (args.latent_dim, 2)
    else:
        encoder_output_shape = (args.latent_dim, )

    if args.encoder == "dense":
        encoder = networks.dense.build_model((concatenated_input_size, ),
                                             encoder_output_shape,
                                             args.encoder_dims,
                                             args.encoder_wd,
                                             args.encoder_use_bn,
                                             args.activation,
                                             "linear")
    elif args.encoder == "conv":
        encoder = conv.build_model((concatenated_input_size, ), encoder_output_shape, args.encoder_conv_channels, args.encoder_wd, args.encoder_use_bn, args.activation, "linear")

    elif args.encoder == "conv_deconv":
        encoder =  conv.build_model_conv_encoder((concatenated_input_size,), encoder_output_shape,
                                      args.encoder_conv_channels,
                                      args.encoder_wd,
                                      args.encoder_use_bn,
                                      args.activation, "linear")
    else:
        assert False, "Unrecognized value for encoder: {}".format(args.encoder)
    encoder = Model(inputs=[input_x, input_y], outputs=encoder(merge_model([input_x, input_y]))) # Sequential([merge_model, encoder])

    generator_input_shape = (args.latent_dim + args.y_label_count, )
    if args.generator == "dense":
        generator = networks.dense.build_model(generator_input_shape,
                                               args.original_shape,
                                               args.generator_dims,
                                               args.generator_wd,
                                               args.generator_use_bn,
                                               args.activation,
                                               "linear")
    elif args.generator == "conv":
        generator = conv.build_model(generator_input_shape, args.original_shape, args.generator_conv_channels, args.generator_wd, args.generator_use_bn, args.activation, "linear")
    elif args.generator == "conv_deconv":
        generator = conv.build_model_conv_decoder(generator_input_shape, args.original_shape, args.generator_conv_channels, args.generator_wd, args.generator_use_bn, args.activation, "linear")
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)
    generator = Model(inputs=[input_latent, input_y2], outputs=generator(merge_latent_model([input_latent, input_y2])))

    if args.sampling:
        hidden = encoder([input_x, input_y])
        sampler_model = add_gaussian_sampling(encoder_output_shape, args)
        (z, z_mean, z_log_var) = sampler_model(hidden)
        output = generator([z, input_y])

        loss_features["z_mean"] = z_mean
        loss_features["z_log_var"] = z_log_var
        ae = Model(inputs=[input_x, input_y], outputs=output)
    else:
        z = encoder([input_x, input_y])
        output = generator([z, input_y]) # Sequential([encoder, generator])
        ae = Model(inputs=[input_x, input_y], outputs=output)

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
