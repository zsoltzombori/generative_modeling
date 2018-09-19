from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape

from util import AttrDict
import model_IO
import autoencoder_loss

import networks.dense


def run(args, data):
    (x_train, x_test) = data

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "encoder_log_var", "generator")) <= set(models.keys()), models.keys()
    
    print("Autoencoder architecture:")
    models.ae.summary()

    # get losses
    loss, metrics = autoencoder_loss.loss_factory(args, loss_features)

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
    models.ae.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # TODO specify callbacks
    cbs = []

    # train the autoencoder
    models.ae.fit(x_train, x_train,
                  verbose=args.verbose,
                  shuffle=True,
                  epochs=args.nb_epoch,
                  batch_size=args.batch_size,
                  callbacks = cbs,
                  validation_data=(x_test, x_test)
    )
    
    # save models
    model_IO.save_autoencoder(models, args)

    # display randomly generated images
    vis.displayRandom((10, 10), args, models, "{}/random".format(args.outdir))

    # display one batch of reconstructed images
    vis.displayReconstructed(x_train[:args.batch_size], args, models, "{}/train".format(args.outdir))
    vis.displayReconstructed(x_test[:args.batch_size], args, models, "{}/test".format(args.outdir))


    # # display image interpolation
    # vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix,
    #                   anchor_indices = data_object.anchor_indices, toroidal=args.toroidal)

    # vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
    # vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))



def build_models(args):
    if args.encoder == "dense":
        encoder_prefix = networks.dense.build_model(args.original_shape, args.encoder_dims, args.encoder_wd, args.encoder_use_bn, args.activation)
    else:
        assert False, "Unrecognized value for encoder: {}".format(args.encoder)

    if args.generator == "dense":
        generator_prefix = networks.dense.build_model((args.latent_dim,), args.generator_dims, args.generator_wd, args.generator_use_bn, args.activation)
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    encoder = Sequential()
    encoder.add(encoder_prefix)
    encoder.add(Dense(args.latent_dim))

    generator = Sequential()
    generator.add(encoder_prefix)
    generator.add(Dense(args.original_size))
    generator.add(Reshape(args.original_shape))

    ae = Sequential([encoder, generator])

    modelDict = AttrDict({})
    modelDict.ae = ae
    modelDict.encoder = encoder
    modelDict.encoder_log_var = encoder # TODO
    modelDict.generator = generator

    loss_features = AttrDict({
        # "z_sampled": z,
        # "z_mean": z_mean,
        # "z_log_var": z_log_var,
        # "z_normed": z_normed,
        # "z_projected": z_projected
    })
    return modelDict, loss_features
