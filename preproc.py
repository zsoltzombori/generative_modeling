import numpy as np
import argparse
from random import randint
from tempfile import TemporaryFile

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the NPZ file")
parser.add_argument("output_number", type=int, help="The desired number of outputs")
args = parser.parse_args()
file_location = args.path
output_number = args.output_number

data = np.load(file_location, encoding="latin1")
print(data.keys())

n = len(data['imgs'])
indices = np.array([], dtype=int)
print('Number of input images: ' + str(n))
print('Number of output images: ' + str(output_number))

for x in range(output_number):
    r = randint(0, n)
    a = np.array([r])
    indices = np.append(indices, a, axis=0)

imgs = 255*data['imgs'][indices]
latents_classes = data['latents_classes'][indices]
latents_values = data['latents_values'][indices]

np.savez_compressed("datasets/reduced", metadata=data['metadata'], imgs=imgs, latents_classes=latents_classes, latents_values=latents_values)
