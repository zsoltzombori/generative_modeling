import data
import numpy as np
dataset = data.load("syn-circles", (64,64))
train, test = dataset.get_data(100, 100)

print(np.min(train[0]), np.max(train[0]))
