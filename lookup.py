import numpy as np
import vis
import load_models_IO
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
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

def get_images(indices):
    data = np.load("datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    imgs = data['imgs']
    return imgs[indices]

modelDict = load_models_IO.load_autoencoder(args)
encoder = modelDict.encoder

for o in range(25, 30):
    for s in range(0,2):

        idc = find_indices_shift(2, s, o)
        images = get_images(idc)
        #vis.plotImages(np.expand_dims(images, axis=3), 32, 32, 'pictures/lookup/test

        batch_size = args.batch_size
        images = np.expand_dims(images, axis=3)
        images = images[:(len(images) // args.batch_size * args.batch_size)]
        res = encoder.predict(images, batch_size = batch_size)
        pca = PCA(n_components=2)
        points = pca.fit_transform(res[1])
        x, y = points.T
        colors = [ [(i%32)/32, (i-i%32)/1024, 0] for i in range(1000)]
        name = "pictures/mass/plot_scale" + str(s) + "ori" + str(o)
        plt.scatter(x, y, c=colors)
        plt.savefig(name)
        print(name + ".png is created!")
        plt.close()
