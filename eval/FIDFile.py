import tensorflow as tf
import sys
import numpy as np

import FID as F

def nchw2nhwc(imgs):
    return np.transpose(imgs, [0, 2, 3, 1])

def nhwc2nchw(imgs):
    return np.transpose(imgs, [0, 3, 1, 2])

print ('FID of data of files "', sys.argv[1], '" and "', sys.argv[2], '": ')

gen_data = []
real_data = []
# Default is nhwc:
nchw = False

assert  len(sys.argv) >= 2, 'Invalid number of atguments'

if len(sys.argv) == 3 and (sys.argv[2] == '-NCHW' or sys.argv[2] == '-nchw'):
    nchw = True
if len(sys.argv) == 4 and (sys.argv[3] == '-NCHW' or sys.argv[3] == '-nchw'):
    nchw = True

assert sys.argv[1].endswith('.npy'), 'Input file of generated data must be .npy'
assert sys.argv[2].endswith('.npy'), 'Input file of real data must be .npy'
gen_data = np.load(sys.argv[1])
print( 'Generated data loaded...')
real_data = np.load(sys.argv[2])
print( 'Real data loaded...')

if nchw == False:
    print('Transposing Data...')
    gen_data = nhwc2nchw(gen_data)
    real_data = nhwc2nchw(real_data)
    

splitnum = 1;

print ('Calculating Frechet Inception Distance...')
FIDS = F.FID(gen_data, real_data, splitnum)
print ('FID of data of files "', sys.argv[1], '" and "', sys.argv[2], '": ', FIDS)
