import model_IO
import loss

def run(args, data):
    (x_train, x_test) = data

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "encoder_log_var", "generator")) < set(keys(models)),
    
    print("Autoencoder architecture:")
    models.ae.summary()

    # get losses
    loss, metrics = loss.loss_factory(loss_features, args)

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
    ae.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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



def build_models(args)
