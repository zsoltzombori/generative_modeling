from keras import objectives
from keras import metrics
import keras.backend as K
import tensorflow as tf
import numpy as np

# loss_features is an AttrDict with all sorts of tensors that are different from the input-output
# various models have different mechanisms for populating it
def loss_factory(loss_names, args, loss_features=None, combine_with_weights=True):

    def xent_loss(x, x_decoded):
        loss = args.original_size * objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)

    def mse_loss(x, x_decoded):
        loss = args.original_size * objectives.mean_squared_error(x, x_decoded)
        return K.mean(loss)

    def mae_loss(x, x_decoded):
        loss = args.original_size * objectives.mean_absolute_error(x, x_decoded)
        return K.mean(loss)

    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(loss_features.z_mean), axis=-1)
        return K.mean(loss)

    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - loss_features.z_log_var + K.exp(loss_features.z_log_var), axis=-1)
        return K.mean(loss)

    def binary_crossentropy_loss(x, x_decoded):
        loss = objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)
    def gan_generator_loss(x, x_decoded):
        loss = - K.log(x_decoded)
        return K.mean(loss)
    def accuracy(x, x_decoded):
        acc = metrics.binary_accuracy(x, x_decoded)
        return K.mean(acc)
    def wasserstein_loss(x, x_decoded):
        #x=(2*x-1)
        return K.mean(x*x_decoded)

    def outside_sphere_loss(x, x_decoded):
        d = args.latent_dim
        z = loss_features.center
        e = K.exp(loss_features.evalues)

        # penalizes if axes of ellipsoid are not contained in unit sphere.
        dsquare = K.sum(K.square(z), axis = -1) + K.sum(K.square(e), axis = -1) / d + 2 * K.dot(K.abs(z), K.transpose(e)) / d
        # simplified version, penalizes if bounding box of ellipsoid is not contained in unit sphere.
        # (seemingly equivalent to axes-based version.)
        # dsquare = K.sum((K.abs(z) + e) ** 2, axis=-1)

        # TODO truncation should probably be applied coordinate-wise.
        loss_outside = 10 * K.maximum(0.0, dsquare - 1)
        return K.mean(loss_outside)

    def outside_box_loss(x, x_decoded):
        d = args.latent_dim
        z = loss_features.center
        e = K.exp(loss_features.evalues)

        margin_sum = K.maximum(K.abs(z) + e - 1, 0)
        loss_outside = 10 * K.square(margin_sum)

        return K.mean(loss_outside)

    def ellipse_loss(x, x_decoded):
        d = args.latent_dim
        z = loss_features.center
        e = K.exp(loss_features.evalues)

        # longer than 1 axes are not promoted, helping the outside_*_loss do its job.
        loss_kl = - K.sum(K.minimum(loss_features.evalues, 0))

        return K.mean(loss_kl)
            

    losses = []
    for loss in loss_names:
        losses.append(locals().get(loss))

    if not combine_with_weights:
        return losses
    else:
        weightDict = {}
        for w in args.weights:
            weightDict[w[0]] = w[1]

        def lossFun(x, x_decoded):
            lossValue = 0
            for i in range(len(losses)):
                loss = losses[i]
                lossName = loss_names[i]
                currentLoss = loss(x, x_decoded)
                weight = weightDict.get(lossName, 1.0)
                currentLoss *= weight
                print(lossName, "weight", weight)
                lossValue += currentLoss
            return lossValue
        return lossFun
    

def gp_loss(x, x_decoded, interpolated):
    gradients = K.gradients(x_decoded, interpolated)[0]
    slope = K.sum(K.square(gradients), axis=np.arange(1, len(gradients.shape)))
    gp = K.square(1-K.sqrt(slope))
    return K.mean(gp)
