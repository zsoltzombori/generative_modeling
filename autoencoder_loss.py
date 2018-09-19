from keras import objectives
import keras.backend as K
import tensorflow as tf
import numpy as np


# loss_features is an AttrDict with values populated in autoencoder.build_models
def loss_factory(args, loss_features):

    def xent_loss(x, x_decoded):
        loss = objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)

    def mse_loss(x, x_decoded):
        loss = objectives.mean_squared_error(x, x_decoded)
        return K.mean(loss)

    def mae_loss(x, x_decoded):
        loss = objectives.mean_absolute_error(x, x_decoded)
        return K.mean(loss)


    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(loss_features.z_mean), axis=-1)
        return K.mean(loss)

    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - loss_features.z_log_var + K.exp(loss_features.z_log_var), axis=-1)
        return K.mean(loss)


    loss_names = args.losses
    metric_names = sorted(set(args.metrics + args.losses))

    metrics = []
    for metric in metric_names:
        metrics.append(locals().get(metric))
    losses = []
    for loss in loss_names:
        losses.append(locals().get(loss))

    weightDict = {}
    for w in args.weights:
        weightDict[w[0]] = w[1]
    print("weight dict", weightDict)

    def lossFun(x, x_decoded):
        lossValue = 0
        for i in range(len(losses)):
            loss = losses[i]
            lossName = args.losses[i]
            print(lossName, loss)
            currentLoss = loss(x, x_decoded)
            weight = weightDict.get(lossName, 1.0)
            currentLoss *= weight
            print(lossName, "weight", weight)
            lossValue += currentLoss
        return lossValue
    return lossFun, metrics
