import numpy as np
import os

import matplotlib as mpl
mpl.use('TkAgg')

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


import vis
import model_IO
import params
args = params.getArgs()

# lables in latents_values idx: ('color':0, 'shape':1, 'scale':2, 'orientation':3, 'posX':4, 'posY':5)
# lables in latents_values qnty: ('color':1, 'shape':3, 'scale':6, 'orientation':40, 'posX':32, 'posY':32)
# color is not a parameter, because it's always '1'
def find_index(shape, scale, orientation, posX, posY):
    posY_const = 1
    posX_const = 32 * posY_const
    orientation_const = 32 * posX_const
    scale_const = 40 * orientation_const
    shape_const = 6 * scale_const

    index = shape * shape_const + scale * scale_const + orientation * orientation_const + posX * posX_const + posY * posY_const
    return index

def find_indices_shift(shape, scale, orientation):
    indices = []
    for posY in range(32):
        for posX in range(32):
            indices.append(find_index(shape, scale, orientation, posX, posY))
    return np.array(indices)

def find_indices_lineY(shape, scale, orientation, posY):
    indices = []
    for posX in range(32):
            indices.append(find_index(shape, scale, orientation, posX, posY))
    return np.array(indices)

def find_indices_lineX(shape, scale, orientation, posX):
    indices = []
    for posY in range(32):
            indices.append(find_index(shape, scale, orientation, posX, posY))
    return np.array(indices)

def get_images(indices):
    data = np.load("datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    imgs = data['imgs']
    return imgs[indices]

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# degrees not radians
def angles(curve):
    l, d = curve.shape
    assert l >= 3
    deltas = curve[1:, :] - curve[:-1, :]
    deltas /= np.linalg.norm(deltas, axis=1, keepdims=True)
    scalar_prods = np.sum(deltas[1:, :] * deltas[:-1, :], axis=1)
    assert scalar_prods.shape == (l-2, )
    return np.arccos(scalar_prods.clip(-1, +1)) * 180 / np.pi


# degrees not radians
def straightness(curve):
    return np.mean(angles(curve))


def test_straightness():
    a = np.arange(20.0)[:, np.newaxis]
    a = np.repeat(a, 5, axis=1)
    print("straight line", straightness(a))

    n = 20
    d = np.pi * 2 / n
    a = np.array([[np.cos(i * d), np.sin(i * d)] for i in range(n)])
    print("regular polygon", straightness(a))

    a = np.random.normal(size=(20,5))
    print("random points", straightness(a))


if __name__ == "__main__":
    test_straightness()

modelDict = model_IO.load_autoencoder(args)
encoder = modelDict.encoder


idc = find_indices_shift(0,0,0)
images = get_images(idc)
#vis.plotImages(np.expand_dims(images, axis=3), 32, 32, 'pictures/lookup/test

batch_size = args.batch_size
images = np.expand_dims(images, axis=3)
images = images[:(len(images) // args.batch_size * args.batch_size)]
res = encoder.predict(images, batch_size = batch_size)
pca = PCA(n_components=2)
points = pca.fit_transform(res[1])

if True:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    print(">>>", plt.get_backend())
    pca = PCA(n_components=3)
    p = pca.fit_transform(res[1])
    colors = [ [(i%32)/32, (i-i%32)/1024, 0] for i in range(1000)]

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors)
    mkdir(args.outdir + "/graphs/all")
    name = args.outdir + "/graphs/all/3D"
    plt.savefig(name)
    plt.gcf().clear()
    #plt.show()

sumY = 0
sumaY = 0
sumX = 0
sumaX = 0

for X in range(32):
    rng = range(X, 31*32 + X, 32)
    #print(rng)
    pointsX = points[rng]
    x, y = pointsX.T
    #colors = [ [(i%32)/32, (i-i%32)/1024, 0] for i in range(1000)]
    colors = [ [ 0, 0, i/30] for i in range(31)]
    mkdir(args.outdir + "/graphs/X")
    name = args.outdir + "/graphs/X/lineX" + str(X)
    plt.scatter(x, y, c=colors)
    plt.savefig(name)
    plt.close()
    #print(name + ".png is created!")
    pca2 = PCA(n_components=1)
    line = pca2.fit(pointsX)
    sumX = sumX + pca2.explained_variance_ratio_[0]
    sumaX = sumaX + straightness(pointsX)

print("PCA STRAIGHTNESS X: " + str(sumX/31))
print("ANGULAR STRAIGHTNESS X: " + str(sumaX/31))

for Y in range(31):
    rng = range(Y*32, Y*32+32)
    #print(rng)
    pointsY = points[rng]
    x, y = pointsY.T
    #colors = [ [(i%32)/32, (i-i%32)/1024, 0] for i in range(1000)]
    colors = [ [ i/31, 0, 0] for i in range(32)]
    mkdir(args.outdir + "/graphs/Y")
    name = args.outdir + "/graphs/Y/lineY" + str(Y)
    plt.scatter(x, y, c=colors)
    plt.savefig(name)
    plt.close()
    #print(name + ".png is created!")
    pca2 = PCA(n_components=1)
    line = pca2.fit(pointsY)
    sumY = sumY + pca2.explained_variance_ratio_[0]
    sumaY = sumaY + straightness(pointsY)

print("PCA STRAIGHTNESS Y: " + str(sumY/30))
print("ANGULAR STRAIGHTNESS Y: " + str(sumaY/30))

def two_lines(X, Y):
    #compare
    #X0
    pointsX0 = points[ range(X, 31*32 + X, 32) ]
    colorsX = [ [ 0, 0, 1] for i in range(0, 31)]
    #Y0
    pointsY0 = points[ range(Y*32, Y*32+32) ]
    colorsY = [ [ 1, 0, 0] for i in range(0, 32)]

    colors = np.append(colorsX, colorsY, axis=0)
    pointsXY0 = np.append(pointsX0, pointsY0, axis=0)
    x, y = pointsXY0.T
    mkdir(args.outdir + "/graphs/XY")
    name = args.outdir + "/graphs/XY/line" + str(X) + "X" + str(Y) + "Y"
    plt.scatter(x, y, c=colors)
    plt.savefig(name)
    plt.close()

#two_lines(0,0)
