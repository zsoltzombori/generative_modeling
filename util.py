from keras.models import Model

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
