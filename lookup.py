import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


import vis
import model_IO
import params
args = params.getArgs()

# labels in latents_values idx: ('color':0, 'shape':1, 'scale':2, 'orientation':3, 'posX':4, 'posY':5)
# labels in latents_values qnty: ('color':1, 'shape':3, 'scale':6, 'orientation':40, 'posX':32, 'posY':32)
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

def get_dsprites():
    data = np.load("datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    dsprites = data['imgs']
    return dsprites

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
def straightness_metric(curve):
    return np.mean(angles(curve))


def evr_metric(curve):
    pca = PCA(n_components=1)
    line = pca.fit(curve)
    evr = pca.explained_variance_ratio_[0]
    return evr


# input shape (number of curves, number of points on curve, dimension of space)
def parallelness_metric(curves):
    n, m, d = curves.shape
    pca = PCA(n_components=1)
    directions = []
    for curve in curves:
        line = pca.fit(curve)
        direction = pca.components_
        assert direction.shape == (1, d)
        direction = direction.flatten()
        directions.append(direction)
    directions = np.array(directions)
    # just to be sure:
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    angle_sum = 0
    angle_count = 0
    for i in range(n):
        for j in range(i, n):
            # abs because we compare lines not directions.
            scalar_prod = np.abs(directions[i].dot(directions[j]))
            angle = np.arccos(scalar_prod.clip(-1, +1)) * 180 / np.pi
            angle_sum += angle
            angle_count += 1
    return angle_sum / angle_count


def test_straightness():
    a = np.arange(20.0)[:, np.newaxis]
    a = np.repeat(a, 5, axis=1)
    print("straight line", straightness_metric(a))

    n = 20
    d = np.pi * 2 / n
    a = np.array([[np.cos(i * d), np.sin(i * d)] for i in range(n)])
    print("regular polygon", straightness_metric(a))

    a = np.random.normal(size=(20,5))
    print("random points", straightness_metric(a))


modelDict = model_IO.load_autoencoder(args)
encoder = modelDict.encoder
generator = modelDict.generator
dsprites = get_dsprites()


index1 = find_index(shape=0, scale=0, orientation=0, posX=0, posY=0)
index2 = find_index(shape=0, scale=0, orientation=0, posX=31, posY=0)
index3 = find_index(shape=0, scale=0, orientation=0, posX=0, posY=31)
anchors = dsprites[[index1, index2, index3]]
imageBatch = np.concatenate([anchors, dsprites[:args.batch_size-3]])
imageBatch = np.expand_dims(imageBatch, axis=3)
vis.displayInterp(imageBatch, args, modelDict, gridSize=10, anchor_indices=[0,1,2], name="{}/interp".format(args.outdir))


def sliding_vis_d(coord):
    d = args.latent_dim
    n = args.batch_size
    r = 3
    vs = []
    for j in range(n):
        x = -r + 2.0 * r * j / (n - 1)
        v = np.zeros(d)
        v[coord] = x
        vs.append(v)
    vs = np.array(vs)

    images_gen = generator.predict([vs], batch_size=n)
    vis.plotImages(images_gen, 10, 20, 'pictures/lookup/test-%d' % coord)


def sliding_vis():
    d = args.latent_dim
    for i in range(d):
        sliding_vis_d(i)

# sliding_vis()


def evalutate_2d_grid(dsprites, grid_indices, do_3d_vis):
    images = dsprites[grid_indices]
    #vis.plotImages(np.expand_dims(images, axis=3), 32, 32, 'pictures/lookup/test

    inum = len(images)
    batch_size = args.batch_size
    images = np.expand_dims(images, axis=3)
    images_padded = np.concatenate((images, images), axis=0)
    images_padded = images_padded[:(inum // args.batch_size * args.batch_size + args.batch_size)]
    assert len(images_padded) >= inum
    assert len(images_padded) % args.batch_size == 0

    res_padded = encoder.predict(images_padded, batch_size=batch_size)
    z_sampled, z_mean, z_logvar = res_padded
    # undo padding:
    z_sampled = z_sampled[:inum] ; z_mean = z_mean[:inum] ; z_logvar = z_logvar[:inum]
    assert len(z_mean) == inum

    # for i in range(args.latent_dim):
    #    print(i, np.std(z_mean[:32, i]), np.std(z_mean[::32, i]))
    #    print(i, np.std(z_mean[16*32:17*32, i]), np.std(z_mean[16::32, i]))

    if do_3d_vis:
        print("VERY SLOW, DON'T RUN IT IN INNER LOOP")
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(z_mean)
        colors = [ [(i%32)/32, (i-i%32)/1024, 0] for i in range(1024)]

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors)
        mkdir(args.outdir + "/graphs/all")
        name = args.outdir + "/graphs/all/3D"
        plt.savefig(name)
        plt.gcf().clear()

        y_slice = z_mean[16::32, :]
        y_slice_01 = y_slice[:, [0, 1]]
        plt.scatter(y_slice_01[:, 0], y_slice_01[:, 1])
        plt.xlim(-2, 2) ; plt.ylim(-2, 2)
        plt.savefig(args.outdir + "/graphs/all/slice_y16_coords01")
        plt.gcf().clear()
        y_slice_23 = y_slice[:, [2, 3]]
        plt.scatter(y_slice_23[:, 0], y_slice_23[:, 1], )
        plt.xlim(-2, 2) ; plt.ylim(-2, 2)
        plt.savefig(args.outdir + "/graphs/all/slice_y16_coords23")
        plt.gcf().clear()

        n, d = y_slice.shape
        f, axes = plt.subplots(d, sharex=True, sharey=True)
        f.set_figheight(15)
        axes[0].set_title('Subplots are coordinates of latent space, swiping through a y slice')
        for i in range(d):
            axes[i].plot(range(n), y_slice[:, i])
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.savefig(args.outdir + "/graphs/all/slice_y16_all_coords")
        plt.gcf().clear()


    straightness_sum = 0.0
    evr_sum = 0.0

    for X in range(32):
        rng = range(X, 32*32, 32)
        z_line = z_mean[rng]
        evr_sum += evr_metric(z_line)
        straightness_sum += straightness_metric(z_line)

    evr_mean = evr_sum / 32
    straightness_mean = straightness_sum / 32

    parallelness = parallelness_metric(z_mean.reshape((32, 32, -1)))

    return evr_mean, straightness_mean, parallelness

grid_indices = find_indices_shift(shape=1, scale=3, orientation=20)
evr, straightness, parallelness = evalutate_2d_grid(dsprites, grid_indices, do_3d_vis=True)
print("Metrics for 3d vis slice %f %f %f" % (evr, straightness, parallelness))

np.random.seed(1337)
planar_slice_specs = []
for i in range(30):
    shape = np.random.randint(3)
    scale = np.random.randint(6)
    orientation = np.random.randint(40)
    planar_slice_specs.append((shape, scale, orientation))

evrs, straightnesses, parallelnesses = [], [], []
for (shape, scale, orientation) in planar_slice_specs:
    grid_indices = find_indices_shift(shape, scale, orientation)
    evr, straightness, parallelness = evalutate_2d_grid(dsprites, grid_indices, do_3d_vis=False)
    print("evr straightness parallelness: %f %f %f" % (evr, straightness, parallelness))
    evrs.append(evr)
    straightnesses.append(straightness)
    parallelnesses.append(parallelness)

print("evr straightness parallelness means: %f %f %f" %
    tuple(map(lambda a: np.mean(np.array(a)), (evrs, straightnesses, parallelnesses))))

plt.scatter(evrs, straightnesses)
mkdir(args.outdir + "/graphs")
plt.savefig(args.outdir + "/graphs/evr-vs-straightness.png")
plt.close()
plt.scatter(evrs, parallelnesses)
plt.savefig(args.outdir + "/graphs/evr-vs-parallelness.png")



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
