import seaborn as sns
import pandas as pd
import numpy as np


import json    # or `import simplejson as json` if on Python < 2.6


fi = open('in.couts.2', 'r')
objs = []
for line in fi:
   obj = json.loads(line)
   objs.append(obj)

data = pd.DataFrame(objs)
print(data)

result = data.pivot(index='size_coeff', columns='var_coeff', values='accuracy')

sns.set()
ax = sns.heatmap(result, annot=True, annot_kws={"size": 6})
ax.get_figure().savefig("x.png")