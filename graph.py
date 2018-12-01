import os
from matplotlib import pyplot as plt
import numpy as np

def get_losses(dir):
    with open("pictures/" + dir + "/main.log","r") as file:
        data=file.read()
        lines=data.split("\n")
        lines=[line for line in lines if "val_mse_loss" in line]
        ls=lines[-1].split(" ")
        losses=[ls[5],ls[8],ls[11],ls[14],ls[17],ls[20],ls[23],ls[26]]
        return losses

def get_metrics(dir):
    with open("pictures/" + dir + "/graphs/main.log","r") as file:
        data=file.read()
        lines=data.split("\n")
        #lines=[line for line in lines if line.find("val_mse_loss")]
        ls=lines[-2].split(" ")
        #print(ls)
        metrics=[ls[4],ls[5],ls[6],ls[7]]
        return metrics

def plot_losses(name):
    dirs = [dI for dI in os.listdir("pictures") if os.path.isdir(os.path.join('pictures',dI)) and name in dI]
    qnt = len(dirs)

    xy = []
    for dir in dirs:
        ds=dir.split("_")
        param=ds[-1]
        xy.append((float(param),float(get_losses(dir)[0])))
        #xy.append((float(param),float(get_metrics(dir)[0])))
        print(xy[-1])

    xy.sort()
    xy = np.array(xy)
    plt.scatter(xy[:,0], xy[:,1])
    #plt.xscale("log")
    plt.savefig(name + "-loss")
    plt.close()

plot_losses("vae_ldim")
