import numpy as np

def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

def spherical_sampler(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample, axis=1, keepdims=True)
    return z_sample

def toroidal_sampler(batch_size, latent_dim):
    assert latent_dim % 2 == 0
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    l2 = np.sqrt(z_sample[:, 0::2] ** 2 + z_sample[:, 1::2] ** 2)
    l22 = np.zeros_like(z_sample)
    # must be a nicer way but who cares
    l22[:,0::2] = l2
    l22[:,1::2] = l2
    z_sample /= l22
    return z_sample


def sampler_factory(args):
    return gaussian_sampler
