import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 1.5})

import model_IO
import params
import params_parse
import data
import pandas as pd

import pickle
import gzip
import tensorflow as tf
import keras.backend as K
from random import randint
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.optimizers import Adam

# Load dataset
dataset_zip = np.load('/home/csadrian/disent/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]
print(latents_classes)
print('Metadata: \n', metadata)


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


def sample_pair_with_fixed_factor(factor_index=None, factor_value=None):
    if factor_index is None:
        factor_index = np.random.randint(1, 6)
        factor_value = np.random.choice(latents_classes[:, factor_index])
    
    filtered = latents_classes[:, factor_index] == factor_value
    
    ambiguous = True
    while ambiguous:
        idx_pair = np.random.choice(np.where(filtered)[0], 2)
        pair = latents_classes[idx_pair]
        same = np.equal(pair[0], pair[1])
        same[factor_index] = False
        same[0] = False # The first feature is always the same. This is just for comparison.
        ambiguous = np.any(same)
    return (idx_pair, factor_index)


def generate_disent_dataset(n=60000):
    pairs = []
    targets = []

    # With dsprites, the probability of generating matching pairs is very low,
    # and the effect of this is very minor, nevertheless we go with unique pairs. 
    seen = set()
    count = 0
    while count < n:
        row = sample_pair_with_fixed_factor()
        pair_set = frozenset(row[0])
        if pair_set not in seen:
            pairs.append(row[0])
            targets.append(row[1])
            seen.add(pair_set)
            count += 1
        if count % 1000 == 0:
            print('{}/{}'.format(count, n))

    pairs = np.array(pairs)
    targets = np.array(targets) - 1 # Classes will be indexed from zero
 
    return (pairs, targets)


def save_disent_dataset():
    pairs, targets = generate_disent_dataset(60000)
    np.savez("pairs_60k.npz", pairs=pairs, targets=targets)
    print("pairs_60k.npz saved.")


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def diagnostic_visualization(images_a, images_b):
    num_images=60
    ncols = 2
    nrows = 30
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows *3, ncols * 3))
    axes = axes.flatten()

    imgs_ = np.concatenate([images_a[:30], images_b[:30]], axis=0)

    for ax_i, ax in enumerate(axes):
      if ax_i < num_images:
        ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
      else:
        ax.axis('off')
    print(pairs_targets[:30])
    plt.savefig("inspect_pairs.png")


def main():

    pairs_filepath = 'pairs_60k.npz'
    if os.path.isfile(pairs_filepath) == False:
        print("Generating pairs differing in one fixed generating factor. This may take a few minutes...")
        save_disent_dataset()

    dataset = np.load(pairs_filepath)
    pairs_indices = dataset['pairs']
    pairs_targets = dataset['targets']
    
    images_a = imgs[pairs_indices[:,0], :]
    images_b = imgs[pairs_indices[:,1], :]

    images_a = np.expand_dims(images_a, axis=-1)
    images_b = np.expand_dims(images_b, axis=-1)

    diff = Input(shape=(10, ))
    out = Dense(5)(diff)
    out = Activation("softmax")(out)
    model = Model(inputs=diff, outputs=out)
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    results = []

    for file in os.listdir("./ini/auto-generated"):
        if file.endswith(".ini"):
            ini_filepath = os.path.join("ini/auto-generated", file)
            args = params_parse.paramsFromIni(ini_filepath)

            reset_weights(model)

            modelDict = model_IO.load_autoencoder(args)
            encoder = modelDict.encoder
            generator = modelDict.generator
    
            z_sampled_a, z_mean_a, z_logvar_a = encode(encoder, images_a, args.batch_size)
            z_sampled_b, z_mean_b, z_logvar_b = encode(encoder, images_b, args.batch_size)

            z_diff_ab = np.abs(z_sampled_a - z_sampled_b)
            z_diff_ab_train = z_diff_ab[:50000]
            z_diff_ab_test = z_diff_ab[50000:]
            pairs_targets_train = pairs_targets[:50000]
            pairs_targets_test = pairs_targets[50000:]

            model.fit(x=z_diff_ab_train, y=pairs_targets_train, batch_size=100, epochs=10, shuffle=True)

            res = model.evaluate(z_diff_ab_test, pairs_targets_test)
            accuracy = res[1]
            weights_dict = {}
            for weights in args.weights:
                weights_dict[weights[0]] = weights[1]
            size_coeff = weights_dict['size_loss']
            var_coeff = weights_dict['variance_loss']

            res = {'size_coeff': size_coeff, 'var_coeff': var_coeff, 'accuracy': accuracy}
            results.append(res)
            print(res)

    print(results)
    with open('res.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()
