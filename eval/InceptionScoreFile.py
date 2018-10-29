import sys
import tensorflow as tf
import numpy as np
import InceptionScore as IS

def nchw2nhwc(imgs):
    return np.transpose(imgs, [0, 2, 3, 1])

def nhwc2nchw(imgs):
    return np.transpose(imgs, [0, 3, 1, 2])

gen_data = []
real_data = []
# Default is nhwc:
nchw = False
is_real = False;

assert  len(sys.argv) == 2 or len(sys.argv) == 3 or len(sys.argv) == 4, 'Invalid number of atguments'

if len(sys.argv) == 3 and (sys.argv[2] == '-NCHW' or sys.argv[2] == '-nchw'):
    nchw = True
if len(sys.argv) == 4 and (sys.argv[3] == '-NCHW' or sys.argv[3] == '-nchw'):
    nchw = True

assert sys.argv[1].endswith('.npy'), 'Input file must be .npy'
gen_data = np.load(sys.argv[1])
print( 'Generated data loaded...')
# if there is a second file:
if len(sys.argv) == 4 or (len(sys.argv) == 3 and nchw == False) :
    assert sys.argv[2].endswith('.npy'), 'Input file must be .npy'
    real_data = np.load(sys.argv[2])
    print( 'Real data loaded...')
    is_real = True;

splitnum = 4

if nchw == False:
    print('Transposing gen_data')
    gen_data = nhwc2nchw(gen_data)

    # slicing the arrays -> REMOVE IT LATER:
#gen_data = gen_data[:200]
#real_data = gen_data
#is_real = True
#nchw = True

#print ('Shape: ' , gen_data.shape)

if is_real == False:
    print('Calculating Inception Score...')
    IScore, ISDev = IS.InceptionScore(gen_data, splitnum)
    print('Inception Score of data in file ', sys.argv[1], ': ',  IScore, 
          '. Deviation: ',  ISDev)
else:
    if nchw == False:    
        print('Transposing real_data')
        real_data = nhwc2nchw(real_data)
    print('Calculating Mode Score...')
    ISScore, ISDev = IS.ModeScore(gen_data, real_data, splitnum)
    print('Mode Score of data in files ',sys.argv[1] ,' (real data in file ',
          sys.argv[2], '): ', IScore, ', ', ISDev)
