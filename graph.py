import os
from matplotlib import pyplot as plt
import numpy as np

def get_losses(dir):
    with open("pictures/" + dir + "/main.log","r") as file:
        data=file.read()
        lines=data.split("\n")
        lines=[line for line in lines if line.find("val_mse_loss")]
        ls=lines[-1].split(" ")
        print(len(lines))
        losses=[ls[5],ls[8],ls[11],ls[14],ls[17],ls[20],ls[23],ls[26]]
        return losses

def plot_losses(name):
    dirs = [dI for dI in os.listdir("pictures") if os.path.isdir(os.path.join('pictures',dI)) and name in dI]
    qnt = len(dirs)

    x=[]
    y=[]
    for dir in dirs:
        ds=dir.split("_")
        param=ds[-1]
        x.append(float(param))
        y.append(float(get_losses(dir)[0]))

    plt.scatter(np.array(x), np.array(y))
    plt.savefig(name + "-loss")
    plt.close()

plot_losses("vae_ldim")
