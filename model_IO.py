from util import AttrDict
from keras.models import model_from_json

def loadModel(filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    jFile = open(jsonFile, 'r')
    loaded_model_json = jFile.read()
    jFile.close()
    mod = model_from_json(loaded_model_json)
    mod.load_weights(weightFile)
    print("Loaded model from files {}, {}".format(jsonFile, weightFile))
    return mod

def saveModel(mod, filePrefix):
    weightFile = filePrefix + ".h5"
    mod.save_weights(weightFile)
    jsonFile = filePrefix + ".json"
    with open(filePrefix + ".json", "w") as json_file:
        json_file.write(mod.to_json())
    print("Saved model to files {}, {}".format(jsonFile, weightFile))
    

def load_autoencoder(args):
    outdir = args.outdir
    modelDict = AttrDict({})
    modelDict.ae = loadModel(outdir + "/ae")
    modelDict.encoder = loadModel(outdir + "/encoder")
    modelDict.generator = loadModel(outdir + "/generator")
    return modelDict
def load_gan(args):
    outdir = args.outdir
    modelDict = AttrDict({})
    modelDict.discriminator = loadModel(outdir + "/discriminator")
    modelDict.gen_disc = loadModel(outdir + "/gen_disc")
    modelDict.generator = loadModel(outdir + "/generator")
    return modelDict


def save_autoencoder(modelDict, args):
    outdir = args.outdir
    saveModel(modelDict.ae, (outdir + "/ae"))
    saveModel(modelDict.encoder, (outdir + "/encoder"))
    saveModel(modelDict.generator, (outdir + "/generator"))
def save_gan(modelDict, args):
    outdir = args.outdir
    saveModel(modelDict.discriminator, (outdir + "/discriminator"))
    saveModel(modelDict.generator, (outdir + "/generator"))
    saveModel(modelDict.gen_disc, (outdir + "/gen_disc"))

