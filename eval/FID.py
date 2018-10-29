import tensorflow as tf
from scipy import linalg
import numpy as np

import InceptionScore as IN


BATCH_SIZE = 100

def calculate_FID(gm, gc, rm, rc, num_corr = 1e-6):
    # setting input dimension to minimal
    gm = np.atleast_1d(gm)
    rm = np.atleast_1d(rm)
    gc = np.atleast_2d(gc)
    rc = np.atleast_2d(rc)
    # assertions: 
    assert gm.shape == rm.shape, \
        'Training and test mean vectors have different lengths'
    assert gc.shape == rc.shape, \
        'Training and test covariances have different dimensions'
    diff = gm -rm
    covmean, _ = linalg.sqrtm(gc.dot(rc), disp=False)
    # to ensure numeric stability:
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % num_corr
        print(msg)
        offset = np.eye(gc.shape[0]) * num_corr
        covmean = linalg.sqrtm((gc + offset).dot(rc + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
        # calculating the actual score: 
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(gc) + np.trace(rc) - 2* tr_covmean)


# real_imgs and gen_imgs
# returns: the Frechet Inception Distance between two distributions
def FID(gen_imgs, real_imgs, splitnum = 1):
    logits = IN.GetInceptionLogits(splitnum)
    real_preds = IN.RunInceptionNetwork(real_imgs, logits, BATCH_SIZE)
    gen_preds = IN.RunInceptionNetwork(gen_imgs, logits, BATCH_SIZE)
    real_mean = np.mean(real_preds)
    gen_mean = np.mean(gen_preds)
    real_cov = np.cov(real_preds)
    gen_cov = np.cov(gen_preds)
    return calculate_FID(gen_mean, gen_cov, real_mean, real_cov)
    

