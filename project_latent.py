import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import vis
import model_IO
import params
import data


def get_data(args):
    data_object = data.load(args.dataset, shape=args.shape, color=args.color)
    (x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
    args.original_shape = x_train.shape[1:]
    args.original_size = np.prod(args.original_shape)
    return x_train, x_test


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize(images):
    inum = len(images)
    batch_size = args.batch_size
    images_padded = np.concatenate((images, images), axis=0)
    images_padded = images_padded[:(inum // args.batch_size * args.batch_size + args.batch_size)]
    assert len(images_padded) >= inum
    assert len(images_padded) % args.batch_size == 0

    res_padded = encoder.predict(images_padded, batch_size=batch_size)
    z_sampled, z_mean, z_logvar = res_padded
    # undo padding:
    z_sampled = z_sampled[:inum] ; z_mean = z_mean[:inum] ; z_logvar = z_logvar[:inum]
    assert len(z_mean) == inum
    print(z_mean[:10])
    print(z_logvar[:10])
    print(z_sampled[:10])
    n = None
    plt.scatter(z_mean[:n, 0], z_mean[:n, 1], c="red")
    plt.scatter(z_mean[:n, 0]+np.exp(z_logvar[:n, 0]), z_mean[:n, 1]+np.exp(z_logvar[:n, 1]), c="blue")
    plt.scatter(z_sampled[:n, 0], z_sampled[:n, 1], c="green")
    plt.savefig("vis.png")

args = params.getArgs()
x_train, x_test = get_data(args)

modelDict = model_IO.load_autoencoder(args)
encoder = modelDict.encoder
generator = modelDict.generator

images = x_train[:1000]
print(">>>", images.shape)
visualize(images)
