import sys
sys.path.insert(0, './eval')
from eval.FID import FID
import os
import model_IO
import params_parse
import numpy as np
import pickle
from evaluate_model import get_data

ini_filepath = os.path.join("ini/auto-generated", os.listdir("./ini/auto-generated")[0])
args = params_parse.paramsFromIni(ini_filepath)

x_train, x_test = get_data(args)

real_imgs = np.transpose(np.repeat(x_test[:], 3, 3), [0, 3, 1, 2])

with open('seen_inis.txt', 'r') as f:
    seen_inis = f.read()

fis_results = []

count_inis = 0

for filename in os.listdir("./ini/auto-generated"):
    if count_inis == 50:
        break
    elif filename.endswith(".ini") and filename not in seen_inis:
        count_inis += 1
        ini_filepath = os.path.join("ini/auto-generated", filename)
        args = params_parse.paramsFromIni(ini_filepath)

        modelDict = model_IO.load_autoencoder(args)
        encoder = modelDict.encoder
        generator = modelDict.generator
        z_sampled = np.random.normal(size=(x_test.shape[0], args.latent_dim))
        decoded_imgs = generator.predict(z_sampled, batch_size=args.batch_size)

        gen_imgs = np.transpose(np.repeat(decoded_imgs[:], 3, 3), [0, 3, 1, 2])
        # print('gen imgs shape: ', gen_imgs.shape)

        weights_dict = {}
        for weights in args.weights:
            weights_dict[weights[0]] = weights[1]
        size_coeff = weights_dict['size_loss']
        var_coeff = weights_dict['variance_loss']

        print('Calculating FID ... ')
        fis = FID(real_imgs, gen_imgs)
        result_dict = {'size_coeff': size_coeff, 'var_coeff': var_coeff, 'fis': fis}
        fis_results.append(result_dict)

        print('current result: ', result_dict)

        with open('seen_inis.txt', 'a') as f:
            f.write(filename)

with open('fis_results.pkl', 'wb') as f:
    pickle.dump(fis_results, f, pickle.HIGHEST_PROTOCOL)
