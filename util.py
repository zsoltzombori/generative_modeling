from keras.models import Model
from keras.layers.merge import _Merge
from keras import backend as K

# Causes memory leak below python 2.7.3
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def print_model(model_or_layer):
    if type(model_or_layer) == Model:
        model_or_layer.summary()
        for layer in model_or_layer.layers:
            print_model(layer)
    else:
        print(model_or_layer.input_shape, " ---> ", model_or_layer.output_shape)

# taken from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        super(RandomWeightedAverage, self).__init__()

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
