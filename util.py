# Causes memory leak below python 2.7.3
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# print inputs and outputs of layers of a model
def print_model_shapes(model):
    for layer in model.layers:
        print(layer.input_shape, " ---> ", layer.output_shape)
