import numpy as np
import keras
import tensorflow as tf

import params
import data

import autoencoder

# load parameters
args = params.getArgs()
print(args)

# limit memory usage
print("Keras version: ", keras.__version__)
if keras.backend._BACKEND == "tensorflow":
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))

# load data
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
args.original_shape = x_train.shape[1:]
args.original_size = np.prod(args.original_shape)


if args.model_type == "autoencoder":
    autoencoder.run(args, (x_train, x_test))
else:
    assert False, "Unrecognized model_type: {}".format(args.model_type)


