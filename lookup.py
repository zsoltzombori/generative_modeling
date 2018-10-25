import numpy as np


# lables in latents_values: (0'color', 1'shape', 2'scale', 3'orientation', 4'posX', 5'posY')
def get_image(shape, scale, orientation, posX, posY):
    data = np.load("datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    labels = data['latents_values']
    print(labels[00])
    print(labels[1])

    c = 0
    for i in range(len(labels)):
        if labels[i][1] == shape and labels[i][2] == scale:
            #print(i)
            c = c + 1
    print(c)

get_image(2, 0.5, 0, 0, 0)
