import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import model_IO
import params
import data


def get_data(args):
    data_object = data.load(args.dataset, shape=args.shape, color=args.color)
    (x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
    args.original_shape = x_train.shape[1:]
    args.original_size = np.prod(args.original_shape)
    return x_train, x_test


def get_data_wl(args):
    data_object = data.load(args.dataset, shape=args.shape, color=args.color)
    ((x_train, y_train), (x_test, y_test)) = data_object.get_data(args.trainSize, args.testSize)
    args.original_shape = x_train.shape[1:]
    args.original_size = np.prod(args.original_shape)
    return ((x_train, y_train), (x_test, y_test))


# TODO generalize to decoding/reconstruction
# TODO is this whole padding business really necessary?
def encode(encoder, images, batch_size):
    inum = len(images)
    images_padded = np.concatenate((images, images), axis=0) # inefficient when len(images) is large.
    images_padded = images_padded[:(inum // batch_size * batch_size + batch_size)]
    assert len(images_padded) >= inum
    assert len(images_padded) % batch_size == 0

    res_padded = encoder.predict(images_padded, batch_size=batch_size)
    z_sampled, z_mean, z_logvar = res_padded
    # undo padding:
    z_sampled = z_sampled[:inum] ; z_mean = z_mean[:inum] ; z_logvar = z_logvar[:inum]
    assert len(z_mean) == inum
    return z_sampled, z_mean, z_logvar


def visualize(z_sampled, z_mean, z_logvar, images, labels):
    n = None

    inum = len(images)

    ells = [Ellipse(xy = z_mean[i],
                    width = 2 * np.exp(z_logvar[i][0]),
                    height = 2 * np.exp(z_logvar[i][1]))
            for i in range(inum)]     
    
    fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})   
    
    blu = []
    g = [] 
    r = []
    c = []
    m = [] 
    y = []
    bla = []
    o = []
    t = []
    br = []
    
    if(len(labels) != 0):
        for i in range(inum):
            ax.add_artist(ells[i])
            ells[i].set_clip_box(ax.bbox)
            ells[i].set_alpha(0.5)
            if labels[i] == 0:     
                ells[i].set_facecolor('blue')
                blu.append(ells[i])
            elif labels[i] == 1:
                ells[i].set_facecolor('green')
                g.append(ells[i])     
            elif labels[i] == 2:
                ells[i].set_facecolor('red')
                r.append(ells[i])
            elif labels[i] == 3:
                ells[i].set_facecolor('cyan')
                c.append(ells[i])
            elif labels[i] == 4:
                ells[i].set_facecolor('magenta')
                m.append(ells[i])
            elif labels[i] == 5:
                ells[i].set_facecolor('yellow')
                y.append(ells[i])
            elif labels[i] == 6:
                ells[i].set_facecolor('black')
                bla.append(ells[i])
            elif labels[i] == 7:
                ells[i].set_facecolor('orange')
                o.append(ells[i])
            elif labels[i] == 8:
                ells[i].set_facecolor('teal')
                t.append(ells[i])
            else:
                ells[i].set_facecolor('brown')
                br.append(ells[i])
            
        ax.legend((blu[0], g[0], r[0], c[0], m[0], y[0], bla[0], o[0], t[0], br[0]), 
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), loc="best")

    else:
        for i in range(inum):
            ax.add_artist(ells[i])
            ells[i].set_clip_box(ax.bbox)
            ells[i].set_alpha(0.5)
            ells[i].set_facecolor('plum')


    
    
    plt.scatter(z_mean[:n, 0], z_mean[:n, 1], s = 1, c="black")

    #plt.scatter(z_mean[:n, 0]+np.exp(z_logvar[:n, 0]), z_mean[:n, 1]+np.exp(z_logvar[:n, 1]), c="blue")
    

    #plt.scatter(z_sampled[:n, 0], z_sampled[:n, 1], s = 1, c="white")

    plt.xlim(-10,10)
    plt.ylim(-10,10)

    plt.savefig("vis.png")


def main():
    args = params.getArgs()

    if(args.dataset == "mnist"):
        ((x_train, y_train), (x_test, y_test)) = get_data_wl(args)
        images = x_train[:1000]
        labels = y_train[:1000]
    else:
        (x_train, x_test) = get_data(args)
        images = x_train[:1000]
        labels = []

    modelDict = model_IO.load_autoencoder(args)
    encoder = modelDict.encoder
    generator = modelDict.generator

    '''
    indices = np.random.choice(len(x_train), 1000, replace=False)
    images = x_train[indices, :]
    labels = y_train[indices]
    '''
    
    z_sampled, z_mean, z_logvar = encode(encoder, images, args.batch_size)

    visualize(z_sampled, z_mean, z_logvar, images, labels)


    '''
    nulla = []
    egy = []
    ketto = []
    harom = []
    negy = []
    ot = []
    hat = []
    het = []
    nyolc = []
    kilenc = []

    for i in range(1000):
        if(labels[i] == 0):
            nulla.append(z_logvar[i])
        if(labels[i] == 1):
            egy.append(z_logvar[i])
        if(labels[i] == 2):
            ketto.append(z_logvar[i])
        if(labels[i] == 3):
            harom.append(z_logvar[i])
        if(labels[i] == 4):
            negy.append(z_logvar[i])
        if(labels[i] == 5):
            ot.append(z_logvar[i])
        if(labels[i] == 6):
            hat.append(z_logvar[i])
        if(labels[i] == 7):
            het.append(z_logvar[i])
        if(labels[i] == 8):
            nyolc.append(z_logvar[i])
        if(labels[i] == 9):
            kilenc.append(z_logvar[i])
    
    print(len(nulla), len(egy))
    
    print(nulla, "nulla")
    print(egy, "egy")
    
    av = []

    for j in range(10):
        a = 0
        for i in range(len(nulla)):
            a += nulla[i][j] / egy[i][j]
        a /= len(nulla)  
        av.append(a)
    
    print(av)
    '''

    m = [0,0]
    num = [0,0]

    for i in range(1000):
        if(z_logvar[i][0] > 0):
            print(i, "első")
            num[0] += 1
        if(z_logvar[i][1] > 0):
            print(i, "második")
            num[1] += 1
    print(num)

    print(np.max(z_logvar, axis=0))
    print(np.mean(z_logvar, axis=0))


if __name__ == "__main__":
    main()
